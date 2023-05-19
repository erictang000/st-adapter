# based on https://github.com/salesforce/pytorch-qrnn
import torch
from torch import nn
import math
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


kernel = '''
extern "C"
__global__ void recurrent_forget_mult(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc

     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging

     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = f[i] * x[i];
     dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
  }
}

extern "C"
__global__ void bwd_recurrent_forget_mult(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gx[i]           = f[i] * running_f;
     // Gradient of F
     gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = running_f - f[i] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}
'''


class GPUForgetMult(torch.autograd.Function):
    configured_gpus = {}
    ptx = None

    @staticmethod
    def compile():
        if GPUForgetMult.ptx is None:
            program = Program(kernel, 'recurrent_forget_mult.cu')
            GPUForgetMult.ptx = program.compile()

        if torch.cuda.current_device() not in GPUForgetMult.configured_gpus:
            m = function.Module()
            m.load(bytes(GPUForgetMult.ptx.encode()))

            GPUForgetMult.forget_mult = m.get_function('recurrent_forget_mult')
            GPUForgetMult.bwd_forget_mult = m.get_function('bwd_recurrent_forget_mult')

            Stream = namedtuple('Stream', ['ptr'])
            GPUForgetMult.stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

            GPUForgetMult.configured_gpus[torch.cuda.current_device()] = (GPUForgetMult.forget_mult, GPUForgetMult.bwd_forget_mult, GPUForgetMult.stream)

        GPUForgetMult.forget_mult, GPUForgetMult.bwd_forget_mult, GPUForgetMult.stream = GPUForgetMult.configured_gpus[torch.cuda.current_device()]

    @staticmethod
    def forward(ctx, f, x, hidden_init=None):
        GPUForgetMult.compile()
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default
        if hidden_init is not None: result[0, :, :] = hidden_init
        else: result = result.zero_()
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.forget_mult(grid=grid, block=(grid_hidden_size, 1), args=[result.data_ptr(), f.data_ptr(), x.data_ptr(), seq_size, batch_size, hidden_size], stream=GPUForgetMult.stream)
        ctx.save_for_backward(f, x, hidden_init, result)
        return result[1:, :, :]

    @staticmethod
    def backward(ctx, grad_h):
        GPUForgetMult.compile()
        f, x, hidden_init, h = ctx.saved_tensors
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_x = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        GPUForgetMult.bwd_forget_mult(grid=grid, block=(grid_hidden_size, 1), args=[h.data_ptr(), f.data_ptr(), x.data_ptr(), grad_h.data_ptr(), grad_f.data_ptr(), grad_x.data_ptr(), grad_h_init.data_ptr(), seq_size, batch_size, hidden_size], stream=GPUForgetMult.stream)
        ###
        if hidden_init is not None:
            return grad_f, grad_x, grad_h_init
        return grad_f, grad_x


class CPUForgetMult(nn.Module):
    @staticmethod
    def forward(f, x, hidden_init=None):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        ###
        return torch.stack(result)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Default: 1.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, lookback_window=1, lookahead_window=0, output_gate=False):
        super().__init__()

        if output_gate: raise NotImplementedError()

        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self.hidden_size = hidden_size if hidden_size else input_size
        self.output_gate = output_gate

        self.conv1d_f = nn.Conv1d(
            in_channels=input_size,
            out_channels=self.hidden_size,
            kernel_size=lookback_window + 1 + lookahead_window,
            stride=1,
        )   # expects batch_size, in_channels, seq_len
        self.conv1d_z = nn.Conv1d(
            in_channels=input_size,
            out_channels=self.hidden_size,
            kernel_size=lookback_window + 1 + lookahead_window,
            stride=1,
        )   # expects batch_size, in_channels, seq_len

        self.gelu = QuickGELU()

    def forward(self, X, hidden=None):
        # X [seq_len, batch_size, input_size]
        seq_len, batch_size, in_channels = X.size()
        X_pad = nn.functional.pad(
            X,
            (0, 0, 0, 0, self.lookback_window, self.lookahead_window),
            "constant",
            0
        )     # => [seq_len+(kernel_size-1), batch, input_size]
        X_pad = X_pad.permute(1, 2, 0)   #=> [batch, input_size, seq_len+(kernel_size-1)]

        # Convert the tensor back to (seq_len, batch, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            raise NotImplementedError()
            Z, F, O = Y.chunk(3, dim=2)
        else:
            # compute the output logits
            Z = self.conv1d_z(X_pad)                # => [batch, hidden_size, seq_len]
            Z = Z.permute(2, 0, 1)              # => [seq_len, batch, hidden_size]
            F = self.conv1d_f(X_pad)                # => [batch, hidden_size, seq_len]
            F = F.permute(2, 0, 1)              # => [seq_len, batch, hidden_size]

        ###
        Z = self.gelu(Z)      # [seq_len, batch, hidden_size]
        F = torch.sigmoid(F)     # [seq_len, batch, hidden_size]

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        if X.is_cuda:
            if hidden is None:
                C = GPUForgetMult.apply(F, Z)
            else:
                C = GPUForgetMult.apply(F, Z, hidden)
        else:
            C = CPUForgetMult.forward(F, Z, hidden)

        # Apply (potentially optional) output gate
        if self.output_gate:
            raise NotImplementedError()
            H = torch.sigmoid(O) * C
        else:
            H = C

        return H, C[-1:, :, :]

class QRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super().__init__()

        self.num_directions = 2 if bidirectional else 1

        self.layers = torch.nn.ModuleList(
            [
                QRNNLayer(
                    input_size=input_size if l == 0 else hidden_size,
                    hidden_size=hidden_size,
                    **kwargs
                )
                for l in range(num_layers)
            ]
        )

        if bidirectional:
            self.layers.extend(
                [
                    QRNNLayer(
                        input_size=input_size if l == 0 else hidden_size,
                        hidden_size=hidden_size,
                        **kwargs
                    )
                    for l in range(num_layers)
                ]
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def reset(self):
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i in range(0, self.num_layers * self.num_directions, self.num_directions):
            layer_forward = self.layers[i]
            input_forward, hn_forward = layer_forward(
                input, None if hidden is None else hidden[i]
            )

            if self.bidirectional:
                layer_backward = self.layers[self.num_layers + i]
                input_backward, hn_backward = layer_backward(
                    input.flip(0), None if hidden is None else hidden[self.num_layers + i]
                )
                input_backward = input_backward.flip(0)

                input = torch.cat([input_forward, input_backward], dim=2)
                next_hidden.extend([hn_forward, hn_backward])
            else:
                input = input_forward
                next_hidden.append(hn_forward)

            if i < (self.num_layers * self.num_directions) - self.num_directions:
                input = torch.nn.functional.dropout(
                    input, p=self.dropout, training=self.training, inplace=False
                )

        next_hidden = torch.cat(next_hidden, 0).view(
            self.num_layers * self.num_directions, *next_hidden[0].size()[-2:]
        )

        return input, next_hidden
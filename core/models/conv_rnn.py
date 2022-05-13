import typing
import torch
import torch.nn as nn
from torch.functional import Tensor

class ConvLSTMCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super(ConvLSTMCell, self).__init__()
        """conv_x  has a valid padding by:
        setting padding = kernel_size // 2
        hidden channels for h & c = intermediate_channels
        """
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding=kernel_size // 2,  bias=True
        )
    def forward(self, x:Tensor, state:typing.Tuple) -> typing.Tuple:
        """
        c and h channels = intermediate_channels so  a * c is valid
        """
        c, h = state
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a * c +  g * d
        h = b * torch.tanh(c)
        return c, h

    def _init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> typing.Tuple:
        width, height = spatial_dim
        c = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return c, h

class ConvGRUCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super(ConvGRUCell, self).__init__()
        """
        x dim = 3
        b * h = b & h should have same dim
        hidden channels for h = intermediate_channels
        """
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels * 2,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
        self.conv_y = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,  bias=True
        )
        self.intermediate_channels = intermediate_channels
    def forward(self, x:Tensor, h:Tensor) -> Tensor:
        y = x.clone()
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        y = torch.cat([y, b * h], dim=1)
        y = torch.tanh(self.conv_y(y))
        h = a * h + (1 - a) * y
        return h

    def _init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> Tensor:
        width, height = spatial_dim
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return h

class BaseConvRNN(nn.Module):
    def __init__(self, cell, num_cells:int, in_channels:int, intermediate_channels:int, kernel_size:int, return_sequence=True, return_cell_states=True):
        super(BaseConvRNN, self).__init__()
        self.num_cells = num_cells
        self.kernel_size = kernel_size
        self.intermediate_channels = intermediate_channels
        self.return_sequence = return_sequence
        self.return_cell_states = return_cell_states
        cells = []
        for _ in range(num_cells):
            cells.append(cell(in_channels, intermediate_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def _get_states(self, x:Tensor, state:typing.Tuple=None) -> typing.List:
        b, s, _, w, h = x.shape
        cell_states = []

        for n in range(self.num_cells):
            seq_states = []
            for i in range(s):
                if i == 0 and n == 0: #first cell, first sequence
                    prev_state = self.cells[n]._init_states(b, self.intermediate_channels, (w, h))
                elif  n > 0 and i > 0: #not first cell but sequence > 0
                    prev_state = seq_states[n - 1]
                elif n > 0 and i == 0: #not first cell but starting first sequence
                    prev_state = cell_states[i - 1][-1]
                else:
                    NotImplementedError()
                current_state = self.cells[n](x[:, i, :, :, :], prev_state)
                seq_states.append(current_state)
            cell_states.append(seq_states)

        return cell_states


class ConvLSTM(BaseConvRNN):
    def __init__(self, num_cells:int, in_channels:int, intermediate_channels:int, kernel_size:int, return_sequence:bool=False, return_cell_states:bool=False):
        super(ConvLSTM, self).__init__(
            ConvLSTMCell, num_cells, in_channels,
            intermediate_channels, kernel_size, return_sequence,
            return_sequence
        )
    def forward(self, x:Tensor, state:typing.Tuple=None) -> typing.Tuple:
        states = self._get_states(x)
        if self.return_cell_states and self.return_sequence:
            return states[:][-1], states
        elif not self.return_sequence and not self.return_cell_states:
            return states[-1][-1], None

        return states[-1][-1] , states

class ConvGRU(BaseConvRNN):
    def __init__(self, num_cells:int, in_channels:int, intermediate_channels:int, kernel_size:int, return_sequence:bool=False, return_cell_states:bool=False):
        super(ConvGRU, self).__init__(
            ConvGRUCell, num_cells, in_channels,
            intermediate_channels, kernel_size, return_sequence,
            return_sequence
        )
    def forward(self, input:Tensor, state:typing.Tuple=None) -> typing.Tuple:

        batch_seq = []
        for batch in range(input.size(0)):
            road_graph = input[batch, 0, None, :, :]
            seq = []
            for i in range(1, 11):
                seq.append(torch.concat(
                    (road_graph, input[batch, i, None, :, :], input[batch, 11 + i, None, :, :]), dim=0))
                
            seq.append(torch.concat(
                (road_graph, input[batch, 11, None, :, :], input[batch, 22, None, :, :]), dim=0))
            x = torch.stack(seq)
            # x = x[None, :]
            # x = torch.unsqueeze(x, dim=0)
            batch_seq.append(x)

        x = torch.stack(batch_seq)


        states = self._get_states(x)
        if self.return_cell_states and self.return_sequence:
            return states[:][-1], states
        elif not self.return_sequence and not self.return_cell_states:
            return states[-1][-1] #, None

        return states[-1][-1] , states



# if __name__ == "__main__":
#     x = torch.randn((4, 11 , 3, 256, 256))
#     h= torch.zeros((4, 32, 256, 256))
#     s = ConvGRU(2, 3, 32, 3, return_sequence=False)
#     y = s(x, h)
#     print(y[0][0].shape)

import time
def main():
    model = ConvGRU(2, 3, 32, 3, return_sequence=False).to("cuda:0")

    state = torch.zeros((12, 32, 256, 256))
    for i in range(20):
        inputs = torch.rand((12, 23, 256, 256)).to("cuda:0")
        t = time.time()
        torch.cuda.synchronize()
        output = model(inputs)
        torch.cuda.synchronize()

        print("inf time =", time.time() - t)

    # for o in output:
    #     print(o.shape)
    print(output.shape)

    # try:
    #     import onnx
    #     x = torch.rand((1, 23, 256, 256)).to("cuda:0")
    #     torch.onnx.export(
    #         model,
    #         x,
    #         "UNet_LSTM_Flow.onnx",
    #         export_params=True,
    #         opset_version=11,
    #         do_constant_folding=False,
    #         input_names=['input'],
    #         output_names=['output'])

    #     onnx_model = onnx.load("UNet_LSTM_Flow.onnx")
    #     model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)
    #     onnx.save(model_with_shapes, "UNet_LSTM_Flow_with_shapes.onnx")
    # except:
    #     print("Install ONNX to convert the model!")


if __name__ == "__main__":
    main()

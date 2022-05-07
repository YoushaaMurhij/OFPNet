import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.shape[2] - x1.shape[2]])
        diffX = torch.tensor([x2.shape[3] - x1.shape[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='floor'), diffX - torch.div(diffX, 2, rounding_mode='floor'),
                        torch.div(diffY, 2, rounding_mode='floor'), diffY - torch.div(diffY, 2, rounding_mode='floor')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvLSTMCell(nn. Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self). __init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # guaranteed to remain unchanged during delivery (h,w).
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn. Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                # i gate, f gate, o gate, g gate are calculated together and then split open
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # each timestamp contains two state tensors: h and c

        # concatenate along channel axis # concatenates input tensors with h-state tensors along the channel dimensions
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # i gate, f gate, o gate, g gate together are calculated, and then split open
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g  # c state tensor update
        h_next = o * torch.tanh(c_next)  # h state tensor update

        return h_next, c_next  # outputs the two state tensors of the current timestamp

    def init_hidden(self, batch_size, image_size):
        """
        initial state tensor initialization. the state tensor of the first timestamp is initialized at 0
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim,
                              height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim,
                              height, width, device=self.conv.weight.device)
        return (init_h, init_c)


class ConvLSTM(nn. Module):
    """
    Parameters: Introduction to parameters
    input_dim: Number of channels in input# Enter the number of channels for the tensor
    hidden_dim: Number of hidden channels # h, c The number of channels for the two state tensors, which can be a list
    kernel_size: Size of kernel in convolutions # The size of the convolution kernel, the convolutional kernel size of all layers is the same by default, and it can also be set that the convolutional kernel size of the lstm layer is different
    num_layers: Number of LSTM layers stacked on each other # convolutional layers, which need to be equal to len(hidden_dim).
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
    return_all_layers: Return the list of computations for all layers # Whether to return the h state of all lstm layers
    Note: Will do same padding. # Same convolutional kernel size, same padding size
    Input: Enter an introduction
            A tensor of size [B, T, C, H, W] or [T, B, C, H, W]
    Output: Introduction to output
    two lists are returned: layer_output_list, last_state_list
    List 0: layer_output_list - Single-layer list, each element represents the output h state of a layer of LSTM layer, and the size of each element = [B, T, hidden_dim, H, W]
    Listing 1: last_state_list-- a bilayer list, each element is a binary list [h,c], indicating the output state of the last timestamp of each layer [h,c],h.size=c.size = [B,hidden_dim,H,W]
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
    Example: Usage example
            >> x = torch.rand((32, 10, 64, 128, 128))
            >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(
            kernel_size, num_layers)  # is converted to a list
        hidden_dim = self._extend_for_multilayer(
            hidden_dim, num_layers)  # is converted to a list
        if not len(kernel_size) == len(hidden_dim) == num_layers:  # judge consistency
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            # THE INPUT DIMENSION OF THE CURRENT LSTM LAYER
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        # Concatenates multiple LSTM layers defined into a network model
        self. cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: everything
        None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()  # automatically gets b,h,w information
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
 
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# if __name__ == "__main__":
#     # data = torch.randn((128, 30, 30))
#     # model = ConvLSTM(input_dim=128,
#     #                  hidden_dim=32,
#     #                  kernel_size=[3],
#     #                  num_layers=1,
#     #                  batch_first=True,
#     #                  bias=True,
#     #                  return_all_layers=True)
#     # layer_output_list, last_state_list = model(data)
#     #
#     # last_layer_output = layer_output_list[-1]
#     # last_layer_last_h, last_layer_last_c = last_state_list[-1]
#     #
#     # print(last_layer_output[:, -1, ...] == last_layer_last_h)
#     # print(last_layer_output.shape)
#     x = torch. rand((30, 3, 3, 128, 128))
#     convlstm = ConvLSTM(input_dim=3,
#                         hidden_dim=[16,16,3],
#                         kernel_size=[(3, 3),(5,5),(7,7)],
#                         num_layers=3,
#                         batch_first=True, bias= True, return_all_layers=False)
#     layer_output_list, last_state_list = convlstm(x)

#     last_layer_output = layer_output_list[-1]
#     last_layer_last_h, last_layer_last_c = last_state_list[-1]

#     #h = last_states[0][0]  # 0 for layer index, 0 for h index
#     #print(h)
#     print('last h:', last_layer_last_h. shape)
#     #print(convlstm)
#     #print('output:', output[-1:][0].shape)

class UNet_LSTM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, sequence=True):
        super(UNet_LSTM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sequence = sequence

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        self.cvlstm1 = ConvLSTM(128, 128, [(3, 3)], 1, True, True, False)
        self.cvlstm2 = ConvLSTM(512, 512, [(3, 3)], 1, True, True, False)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input):
        road_graph = input[:, 0, :, :]
        if self.sequence:
            seq = []
            for i in range(1, 11):
                seq.append(torch.concat(
                    (road_graph, input[:, i, :, :], input[:, 11 + i, :, :]), dim=0))
            seq.append(torch.concat(
                (road_graph, input[:, 11, :, :], input[:, 22, :, :]), dim=0))
            input = torch.stack(seq)
            input = torch.unsqueeze(input, dim=0)

        else:
            input = torch.unsqueeze(input, dim=1)

        b, _, _, _, _ = input.shape

        x1, x2, x3, x4 = [], [], [], []

        for i in range(b):
            a = input[i, ...]
            x1.append(self.inc(a))
            x2.append(self.down1(x1[i]))
            x3.append(self.down2(x2[i]))
            x4.append(self.down3(x3[i]))
            # x5.append(self.down4(x4[i]))
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)
        x3 = torch.stack(x3)
        x4 = torch.stack(x4)

        x2_data, x2_target = x2[:, 0:8, ...], x2[:, -8:, ...]
        x2_cl_outs = self.cvlstm1(x2_data)
        x4_data, x4_target = x4[:, 0:8, ...], x4[:, -8:, ...]
        x4_cl_outs = self.cvlstm2(x4_data)
        x4 = x4_target
        b, _, _, _, _ = x4.shape
        logits = []
  
        for i in range(b):
            x = self.up1(x4_cl_outs[0][0][i, ...], x3[i, -8:, ...])
            x = self.up2(x, x2_cl_outs[0][0][i])
            x = self.up3(x, x1[i, -8:, ...])

            logits.append(self.outc(x))
        logits = torch.stack(logits)
        if self.sequence:
            observed, occluded, flow_dx, flow_dy = [], [], [], []
            for i in range(8):
                observed.append(logits[:, i, 0, :, :])
                occluded.append(logits[:, i, 1, :, :])
                flow_dx.append(logits[: , i, 2, :, :])
                flow_dy.append(logits[: , i, 3, :, :])
            logits_list =  observed + occluded + flow_dx + flow_dy
            logits = torch.stack(logits_list)
            logits = torch.permute(logits, (1, 0, 2, 3))
        else:
            logits = torch.squeeze(logits, dim=1)
        
        return logits


def main():
    model = UNet_LSTM(n_channels=3, n_classes=4, sequence=True).to("cuda:0")

    for i in range(10):
        inputs = torch.rand((1, 23, 256, 256)).to("cuda:0")
        t = time.time()
        torch.cuda.synchronize()
        output = model(inputs)

        torch.cuda.synchronize()

        print("inf time =", time.time() - t)
    print(output.shape)

    # try:
    #     import onnx
    #     x = torch.rand((1, 23, 256, 256)).to("cuda:0")
    #     torch.onnx.export(
    #         model,
    #         x,
    #         "UNet_LSTM.onnx",
    #         export_params=True,
    #         opset_version=11,
    #         do_constant_folding=False,
    #         input_names=['input'],
    #         output_names=['output'])

    #     onnx_model = onnx.load("UNet_LSTM.onnx")
    #     model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)
    #     onnx.save(model_with_shapes, "UNet_LSTM_with_shapes.onnx")
    # except:
    #     print("Install ONNX to convert the model!")


if __name__ == "__main__":
    main()

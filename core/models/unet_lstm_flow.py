import sqlite3
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small


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


class UNet_LSTM_Flow(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_LSTM_Flow, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.T = 3

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

        self.flow_model = raft_small(pretrained=True, progress=False)
        # self.flow_model.requires_grad_(False)


    def forward(self, input):
        
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

        x2_data, x2_target = x2[:, 0:self.T, ...], x2[:, -self.T:, ...]
        x2_cl_outs = self.cvlstm1(x2_data)
        x4_data, x4_target = x4[:, 0:self.T, ...], x4[:, -self.T:, ...]
        x4_cl_outs = self.cvlstm2(x4_data)
        x4 = x4_target
        b, _, _, _, _ = x4.shape
        logits = []
  
        for i in range(b):
            x = self.up1(x4_cl_outs[0][0][i, ...], x3[i, -self.T:, ...])
            x = self.up2(x, x2_cl_outs[0][0][i])
            x = self.up3(x, x1[i, -self.T:, ...])

            logits.append(self.outc(x))
        occupancies = torch.stack(logits)
        # occupancies = torch.squeeze(occupancies, dim=1)

        result = []
        for i in range(b):
            merged = []
            for j in range(9):
                merged.append(torch.stack([occupancies[i][0][j], occupancies[i][0][j+ 8], torch.max(occupancies[i][0][j], occupancies[i][0][j + 8])]))
            megred_occupancies = torch.stack(merged)
            megred_occupancies = torch.unsqueeze(megred_occupancies, dim=1)
            # flow_imgs = []
            # for j in range(1, 9):
            #     list_of_flows = self.flow_model(megred_occupancies[j - 1], megred_occupancies[j])
            #     predicted_flows = list_of_flows[-1]
            #     flow_imgs.append(predicted_flows)
            flow_imgs = [self.flow_model(megred_occupancies[j - 1], megred_occupancies[j])[-1] for j in range(1,9) ]
            flows = torch.stack(flow_imgs)
            flows = torch.squeeze(flows, dim=1)
            flows = torch.reshape(flows,(16, 256, 256))
            res = torch.cat((occupancies[i][0][:16], flows),)
            result.append(res)
        # res = torch.squeeze(occupancies, dim=1)
        
        return torch.stack(result)



def main():
    model = UNet_LSTM_Flow(n_channels=23, n_classes=18).to("cuda:0")

    for i in range(20):
        inputs = torch.rand((3, 23, 256, 256)).to("cuda:0")
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

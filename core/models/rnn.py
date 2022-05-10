
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import numpy as np
# ----------------------SpatioTemporal LSTM ------------------------------------------

#   Architecture: PredRNN with combined Static-Semantic objects black and white images


class Network_combinedStaticSemantic(nn.Module):
    def __init__(self, args):
        super(Network_combinedStaticSemantic, self).__init__()
        self.args = args
        self.RNN = RNN(args)

    def forward(self, seq_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = seq_tensor.shape
        seq_tensor = seq_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd)
        next_frames = self.RNN(seq_tensor, numiterations)
        return next_frames

#   Architecture: PredRNN with separate Static and Semantic objects black and white images


class Network_static_Semantic(nn.Module):
    def __init__(self, args):
        super(Network_static_Semantic, self).__init__()
        self.args = args
        self.RNN_static = RNN(args)
        self.RNN_Semantic = RNN(args)

    def forward(self, staticgrid_tensor, Semanticgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        # static grid
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape
        staticgrid_tensor = staticgrid_tensor.contiguous().view(
            batch*seqlen, imgch, imght, imgwd)

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # Semantic grid
        batch, seqlen, imght, imgwd, imgch = Semanticgrid_tensor.shape
        Semanticgrid_tensor = Semanticgrid_tensor.contiguous().view(batch*seqlen,
                                                                    imgch, imght, imgwd)

        next_frames_Semantic = self.RNN_Semantic(
            Semanticgrid_tensor, numiterations)

        return next_frames_static, next_frames_Semantic

#   Architecture: PredRNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images


class Network_static_full(nn.Module):
    def __init__(self, args):
        super(Network_static_full, self).__init__()
        self.args = args
        self.RNN_static = RNN(args)
        self.RNN_full = RNN(args)

    def forward(self, staticgrid_tensor, fullgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape
        staticgrid_tensor = staticgrid_tensor.contiguous().view(
            batch*seqlen, imgch, imght, imgwd)

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # full grid
        batch, seqlen, imght, imgwd, imgch = fullgrid_tensor.shape
        fullgrid_tensor = fullgrid_tensor.contiguous().view(
            batch*seqlen, imgch, imght, imgwd)

        next_frames_full = self.RNN_full(fullgrid_tensor, numiterations)

        return next_frames_static, next_frames_full


# ------------------- STANDARD ConvLSTM --------------------------------------------------------------

#   Architecture: Standard RNN with separate Static and Full scenes as input and predicting separate Static and Semantic objects black and white images
class Network_standard_static_full(nn.Module):
    def __init__(self, args):
        super(Network_standard_static_full, self).__init__()
        self.args = args
        self.RNN_static = Standard_RNN(args)
        self.RNN_full = Standard_RNN(args)

    def forward(self, staticgrid_tensor, fullgrid_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = staticgrid_tensor.shape
        staticgrid_tensor = staticgrid_tensor.contiguous().view(
            batch*seqlen, imgch, imght, imgwd)

        next_frames_static = self.RNN_static(staticgrid_tensor, numiterations)

        # full grid
        batch, seqlen, imght, imgwd, imgch = fullgrid_tensor.shape
        fullgrid_tensor = fullgrid_tensor.contiguous().view(
            batch*seqlen, imgch, imght, imgwd)
        next_frames_full = self.RNN_full(fullgrid_tensor, numiterations)

        return next_frames_static, next_frames_full

#   Architecture: Standard RNN with combined Static-Semantic objects black and white images


class Network_ConvLSTM_combinedStaticSemantic(nn.Module):
    def __init__(self, is_training, device):
        super(Network_ConvLSTM_combinedStaticSemantic, self).__init__()
        self.RNN = Standard_RNN(is_training=is_training, device=device, )

    def forward(self, seq_tensor, numiterations):
        # Passing the sequences and mask tensor to RNN network
        batch, seqlen, imght, imgwd, imgch = seq_tensor.shape
        seq_tensor = seq_tensor.contiguous().view(batch*seqlen, imgch, imght, imgwd)
        next_frames = self.RNN(seq_tensor, numiterations)
        return next_frames


def reverse_schedule_sampling_exp(r_sampling_step1, r_sampling_step2, r_exp_alpha, input_len, batch_size, seq_len, numiterations, imgwd, imght, imgchannels):
    # setting the r_eta
    if numiterations < r_sampling_step1:
        r_eta = 0.5
    elif numiterations < r_sampling_step2:
        r_eta = 1.0 - 0.5 * \
            math.exp(-float(numiterations - r_sampling_step1) /
                     r_exp_alpha)
    else:
        r_eta = 1.0

    # setting the eta
    if numiterations < r_sampling_step1:
        eta = 0.5
    elif numiterations < r_sampling_step2:
        eta = 0.5 - (0.5 / (r_sampling_step2 - r_sampling_step1)
                     ) * (numiterations - r_sampling_step1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, input_len - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, seq_len - input_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((imgwd, imght, imgchannels))
    zeros = np.zeros((imgwd, imght, imgchannels))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(seq_len - 2):
            if j < input_len - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (input_len - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (batch_size, seq_len - 2, imgwd,
                                                   imght, imgchannels))

    return real_input_flag


def schedule_sampling(input_len, batch_size, seq_len, scheduled_sampling, sampling_stop_iter, sampling_changing_rate, eta, numiterations, imgwd, imght, imgchannels):

    zeros = np.zeros((batch_size, seq_len - input_len - 1,
                      imgwd, imght,  imgchannels))

    if not scheduled_sampling:
        return 0.0, zeros

    if numiterations < sampling_stop_iter:
        eta -= sampling_changing_rate  # linear decay (eta_k in the paper)
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (batch_size, seq_len - input_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((imgwd, imght, imgchannels))
    zeros = np.zeros((imgwd, imght, imgchannels))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(seq_len - input_len - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (batch_size, seq_len - input_len - 1, imgwd,
                                                   imght, imgchannels))

    return eta, real_input_flag


def test_mask(reverse_scheduled_sampling, input_len, batch_size, seq_len, imgwd, imght, imgchannels):
    if reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = input_len

    real_input_flag = np.zeros(
        (batch_size, seq_len-mask_input-1, imgwd, imght, imgchannels))

    if reverse_scheduled_sampling == 1:
        real_input_flag[:, :input_len - 1, :, :] = 1.0

    return real_input_flag


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.num_hidden = args.num_hidden
        self.num_layers = len(self.num_hidden)
        cell_list = []

        for i in range(self.num_layers):
            if i == 0:
                in_channel = args.img_channels
            else:
                in_channel = self.args.num_hidden[0]
            num_hidden = self.args.num_hidden[0]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel, num_hidden, self.args.filter_size, self.args.stride)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers-1], args.img_channels,
                                   kernel_size=1, stride=self.args.stride, padding=0, bias=False)

    def forward(self, seq_tensors, numiterations):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        eta = self.args.sampling_start_value
        _, imgchannels, imght, imgwd = seq_tensors.shape
        seq_tensors = seq_tensors.contiguous().view(
            self.args.batch_size, self.args.seq_len, imgchannels, imght, imgwd)

        if self.args.is_training:
            if self.args.reverse_scheduled_sampling == 1:
                mask_true = reverse_schedule_sampling_exp(
                    self.args, numiterations, imgwd, imght, imgchannels)
            else:
                eta, mask_true = schedule_sampling(
                    self.args, eta, numiterations, imgwd, imght, imgchannels)
        else:
            mask_true = test_mask(self.args, imgwd, imght, imgchannels)

        frames = seq_tensors
        mask_true = torch.from_numpy(
            mask_true).contiguous().permute(0, 1, 4, 2, 3)
        mask_true = mask_true.to(self.args.device).float()
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [self.args.batch_size, self.num_hidden[i], imght, imgwd]).to(self.args.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [self.args.batch_size, self.num_hidden[0], imght, imgwd]).to(self.args.device)

        for t in range(self.args.seq_len - 1):
            if self.args.reverse_scheduled_sampling:
                if t == 0:
                    x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - 1] * frames[:, t] +
                           (1 - mask_true[:, t - 1]) * x_gen)

            else:
                if t < self.args.input_len:
                    x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - self.args.input_len] * frames[:, t] +
                           (1 - mask_true[:, t - self.args.input_len]) * x_gen)

            h_t[0], c_t[0], memory = self.cell_list[0](
                x_t, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(
            1, 0, 3, 4, 2).contiguous()

        return next_frames


class Standard_RNN(nn.Module):
    def __init__(self,
                 is_training,
                 device,  
                 num_hidden=[4, 4, 4, 4], 
                 img_channels=3, 
                 filter_size=5, 
                 stride=1, 
                 batch_size=1, 
                 seq_len=11, 
                 reverse_scheduled_sampling=1,
                 sampling_start_value=1.0,
                 input_len=6,
                 r_sampling_step1 = 25000, 
                 r_sampling_step2 = 50000, 
                 r_exp_alpha = 5000,
                 sampling_stop_iter = 50000,
                 sampling_changing_rate =  0.00002,
                 scheduled_sampling = 1,
                 ):
        super(Standard_RNN, self).__init__()
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.is_training = is_training
        self.sampling_start_value = sampling_start_value
        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.device = device
        self.num_layers = len(self.num_hidden)
        self.input_len = input_len
        self.r_sampling_step1 = r_sampling_step1 
        self.r_sampling_step2 = r_sampling_step2 
        self.r_exp_alpha = r_exp_alpha 
        self.sampling_stop_iter = sampling_stop_iter
        self.sampling_changing_rate = sampling_changing_rate
        self.scheduled_sampling = scheduled_sampling
        cell_list = []

        for i in range(self.num_layers):
            if i == 0:
                in_channel = img_channels
            else:
                in_channel = self.num_hidden[0]
            num_hidden = self.num_hidden[0]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden, kernel_size=filter_size)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            self.num_hidden[self.num_layers-1], img_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        
        self.head_in_ch = 30
        self.observed_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.occluded_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.flow_dx_head  = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.flow_dy_head  = sepHead(ch_in=self.head_in_ch, ch_out=8)

    def forward(self, input, numiterations):
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

        seq_tensors = torch.stack(batch_seq)
        # print(seq_tensors.shape)




        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        eta = self.sampling_start_value
        # print(seq_tensors.shape)
        _, _, imgchannels, imght, imgwd = seq_tensors.shape
        seq_tensors = seq_tensors.contiguous().view(
            self.batch_size, self.seq_len, imgchannels, imght, imgwd)

        if self.is_training:
            if self.reverse_scheduled_sampling == 1:
                mask_true = reverse_schedule_sampling_exp(
                    self.r_sampling_step1, self.r_sampling_step2, self.r_exp_alpha, self.input_len, self.batch_size, self.seq_len, numiterations, imgwd, imght, imgchannels)
            else:
                eta, mask_true = schedule_sampling(
                    self.input_len, self.batch_size, self.seq_len, self.scheduled_sampling, self.sampling_stop_iter, self.sampling_changing_rate, eta, numiterations, imgwd, imght, imgchannels)
        else:
            mask_true = test_mask(self.reverse_scheduled_sampling, self.input_len, self.batch_size, self.seq_len, imgwd, imght, imgchannels)

        frames = seq_tensors
        mask_true = torch.from_numpy(
            mask_true).contiguous().permute(0, 1, 4, 2, 3)
        mask_true = mask_true.to(self.device).float()
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [self.batch_size, self.num_hidden[i], imght, imgwd]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.seq_len - 1):
            if self.reverse_scheduled_sampling:
                if t == 0:
                    x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - 1] * frames[:, t] +
                           (1 - mask_true[:, t - 1]) * x_gen)

            else:
                if t < self.input_len:
                    x_t = (frames[:, t])
                else:
                    x_t = (mask_true[:, t - self.input_len] * frames[:, t] +
                           (1 - mask_true[:, t - self.input_len]) * x_gen)

            h_t[0], c_t[0] = self.cell_list[0](x_t, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(
            1, 0, 3, 4, 2).contiguous()


        next_frames = torch.permute(next_frames, (0, 1, 4, 2, 3))
        # print(next_frames.shape)
        next_frames = next_frames.reshape((1, 30, 256, 256))

        out1 = self.observed_head(next_frames)
        out2 = self.occluded_head(next_frames)
        out3 = self.flow_dx_head(next_frames)
        out4 = self.flow_dy_head(next_frames)

        next_frames = torch.cat([out1, out2, out3, out4], dim=1)

        return next_frames
    
class sepHead(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(sepHead,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1,stride=1,bias=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x 


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=num_hidden*7,
                      kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden*4,
                      kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden*3,
                      kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden*2, out_channels=num_hidden,
                      kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        )
        self.conv_last = nn.Conv2d(
            num_hidden*2, num_hidden, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + c_new + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


# Code source: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int   Number of channels of input tensor.
        hidden_dim: int  Number of channels of hidden state.
        kernel_size: (int, int)  Size of the convolutional kernel.
        bias: bool       Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = False

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, h_cur, c_cur):

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


def main():
    device = "cuda:0"
    device = 'cuda:' + str(0)
    # model = Network_ConvLSTM_combinedStaticSemantic(is_training=True, device=device) #.to("cuda:0")
    model = Standard_RNN(is_training=True, device=device).to("cuda:0")
    for i in range(10):
        inputs = torch.rand((1, 23, 256, 256)).to("cuda:0")
        t = time.time()
        torch.cuda.synchronize()
        output = model(inputs, i)

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

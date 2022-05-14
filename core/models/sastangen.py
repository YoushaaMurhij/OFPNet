from torch import nn
from torch.autograd import Variable
import torch
from torch.nn import init
########################################################################
def get_gru_initial_state(num_samples, gru_dim):
    return Variable(torch.FloatTensor(num_samples, gru_dim).normal_())  # m

class FReLU(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1):
        super().__init__()
        self.f_cond = nn.Conv2d(in_c, in_c, kernel_size=k,stride=s, padding=p,groups=in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        tx = self.bn(self.f_cond(x))
        out = torch.max(x,tx)
        return out

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------f
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class Seq2seqGRU(nn.Module):
    def __init__(self, z_dim=100, image_size=256, n_channels=23, gru_dim=128, T=1, out_channels=32, gpu=0):
        self.z_dim = z_dim
        self.image_size = image_size
        self.n_channels = n_channels
        self.gru_dim = gru_dim
        self.T = T
        self.gpu = gpu
        self.out_channels = out_channels
        self.conv = int(self.image_size/16)
        super(Seq2seqGRU, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * self.conv * self.conv, self.z_dim),
            # nn.ReLU(True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.gru_dim, 512 * self.conv * self.conv),
            # nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.gru_dim, self.z_dim),
            nn.ReLU(True),
        )

        self.gru = nn.GRU(self.z_dim, self.gru_dim, batch_first=True)
        # self.grucelldecoder = nn.GRU(z_dim,ngru)

    def forward(self, x):
        bs = x.size(0)
        feature = self.encoder(x.reshape(bs * self.T, self.n_channels, self.image_size, self.image_size))
        z = self.fc1(feature.reshape(bs * self.T, -1))
        z_ = z.reshape(bs, self.T, self.z_dim)
        h = get_gru_initial_state(bs, self.gru_dim).unsqueeze(0).to(self.gpu)
        o, _ = self.gru(z_, h)

        o = self.fc3(o[:,-1])

        xhat = self.decoder(o.reshape(bs, 512, self.conv, self.conv)).reshape(bs, self.out_channels, self.image_size, self.image_size)

        return xhat

class SASTANGen(nn.Module):
    def __init__(self, ch=64, dropout=False, n_channels=3, output_channels=32, lstm_dim=128, image_size=256, T=11):
        self.n_channels = n_channels
        self.lstm_dim   = lstm_dim
        self.image_size = image_size
        self.T = T
        self.output_channels = output_channels


        super(SASTANGen, self).__init__()
        self.enc1 = self.conv_bn_relu(self.n_channels, ch, kernel_size=3,no_batch=True)  # 32x96x96
        self.enc2 = self.conv_bn_relu(ch, ch*2, kernel_size=3, pool_kernel=2)  # 64x24x24
        self.enc3 = self.conv_bn_relu(ch*2, ch*4, kernel_size=3, pool_kernel=2)  # 128x12x12
        self.enc4 = self.conv_bn_relu(ch*4, ch*8, kernel_size=3, pool_kernel=2)  # 256x6x6
        # self.enc5 = self.conv_bn_relu(ch * 8, ch * 8, kernel_size=3, pool_kernel=2)  # 256x6x6
        # self.enc6 = self.conv_bn_relu(ch * 8, ch * 16, kernel_size=3, pool_kernel=2)  # 256x6x6


        # self.dec_1 = self.conv_bn_relu(ch * 16, ch * 8, kernel_size=3, pool_kernel=-2,drop_out=False)  # 256x6x6
        # self.dec0 = self.conv_bn_relu(ch * 8, ch * 8, kernel_size=3, pool_kernel=-2)  # 256x6x6
        self.dec1 = self.conv_bn_relu(ch * 8, ch * 4, kernel_size=3, pool_kernel=-2)  # 128x12x12
        self.dec2 = self.conv_bn_relu(ch * 4 , ch * 2, kernel_size=3, pool_kernel=-2)  # 64x24x24
        self.dec3 = self.conv_bn_relu( ch * 2, ch, kernel_size=3, pool_kernel=-2)  # 32x96x96
        self.dec4 = self.conv_bn_relu(ch  , ch, kernel_size=3)#, pool_kernel=-2)  # 32x96x96
        self.dec5 = nn.Sequential(
            nn.Conv2d(ch ,self.output_channels,  kernel_size=1, padding=0),
            nn.Tanh()
            #nn.Sigmoid()
        )

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=256*2,
                                               hidden_dim=self.lstm_dim,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=self.lstm_dim,
                                               hidden_dim=self.lstm_dim,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_3_convlstm = ConvLSTMCell(input_dim=self.lstm_dim,  # nf + 1
                                               hidden_dim=self.lstm_dim,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_convlstm = ConvLSTMCell(input_dim=self.lstm_dim,
                                               hidden_dim=256*2,
                                               kernel_size=(3, 3),
                                               bias=True)


        self.init_weights()
        # initialize_weights(self)
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.ConvTranspose2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None,no_batch=False,drop_out=False):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        if no_batch:
            layers.append(FReLU(out_ch))
        else:
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(FReLU(out_ch))
            if drop_out:
                nn.Dropout(0.5)
        #layers.append(nn.LeakyReLU(0.2))
        #layers.append(Tanhexp())
        return nn.Sequential(*layers)

    def convlstm_layer(self, x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        for t in range(seq_len):

            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],#[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            h_t3, c_t3 = self.encoder_3_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
        # encoder_vector
        encoder_vector = h_t2
        # decoder
        # for t in range(future_step):
        h_t4, c_t4 = self.decoder_convlstm(input_tensor=h_t3,
                                             cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here

        return h_t4

    def forward(self, input):
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
        # x = torch.unsqueeze(x, dim=0)

        b, seq_len, _, h, w = x.size()
        h = int(2*h/(16))#+2
        w = int(2*w / (16))#+2
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        x1 = self.enc1(x.reshape(b*self.T,self.n_channels,self.image_size,self.image_size))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        # x5 = self.enc5(x4)
        # x6 = self.enc6(x5)
        feature_enc = x4.reshape(b,self.T,256*2,h,w)
        feature_lstm = self.convlstm_layer(feature_enc,self.T, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        # out = self.dec_1(feature_lstm)
        # out = self.dec0(out)
        # out = self.dec1(out)
        out = self.dec1(feature_lstm)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        xhat = self.dec5(out)

        return xhat

import time
def main():
    # model = SASTANGen(n_channels=3, output_channels=32).to("cuda:0")
    model = Seq2seqGRU(n_channels=23, out_channels=32, gpu='cuda:0').to("cuda:0")
    # print(model)
    for i in range(10):
        inputs = torch.rand((16, 23, 256, 256)).to("cuda:0")
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
    #         "SASTANGen.onnx",
    #         export_params=True,
    #         opset_version=11,
    #         do_constant_folding=False,
    #         input_names=['input'],
    #         output_names=['output'])

    #     onnx_model = onnx.load("SASTANGen.onnx")
    #     model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)
    #     onnx.save(model_with_shapes, "SASTANGen_with_shapes.onnx")
    # except:
    #     print("Install ONNX to convert the model!")


if __name__ == "__main__":
    main()

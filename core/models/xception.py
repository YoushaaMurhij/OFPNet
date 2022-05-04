
import timm
import torch
import torch.nn as nn

class Xception(nn.Module):
    def __init__(
        self, model_name, in_channels=23, time_limit=8, n_traj=64, with_head=True,
    ):
        super().__init__()
        
        if with_head:
            self.factor = 4
        else:
            self.factor = 16
        self.n_traj = n_traj
        self.time_limit = time_limit
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes= self.n_traj * self.n_traj * self.time_limit * self.factor,
        )

        self.head_in_ch = 8
        self.observed_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.occluded_head = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.flow_dx_head  = sepHead(ch_in=self.head_in_ch, ch_out=8)
        self.flow_dy_head  = sepHead(ch_in=self.head_in_ch, ch_out=8)


    def forward(self, x):
        x = self.model(x)

        if self.factor == 4:
            x = x.view(-1, 8, 256, 256)

            out1 = self.observed_head(x)
            out2 = self.occluded_head(x)
            out3 = self.flow_dx_head(x)
            out4 = self.flow_dy_head(x)

            logits = torch.cat([out1, out2, out3, out4], dim=1)
        else:
            logits = x.view(-1, 32, 256, 256)

        return logits

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

import time
# import onnx 

def main():
    model = Xception('xception71', in_channels=23, time_limit=8, n_traj=128, with_head=False).to("cuda:0")
    # x = torch.rand((1,23,256,256)).to("cuda:0")
    # torch.onnx.export(
    #     model,
    #     x, 
    #     "R2AttU_sepHead.onnx", 
    #     export_params=True, 
    #     # opset_version=11,
    #     do_constant_folding=False, 
    #     input_names = ['input'], 
    #     output_names = ['output'])

    # onnx_model = onnx.load("R2AttU_sepHead.onnx")
    # model_with_shapes = onnx.shape_inference.infer_shapes(onnx_model)
    # onnx.save(model_with_shapes, "R2AttU_sepHead_with_shapes.onnx")
    
    for i in range(10):
        inputs = torch.rand((1,23,256,256)).to("cuda:0")
        t = time.time()
        torch.cuda.synchronize()
        output = model(inputs)
        torch.cuda.synchronize()

        print("inf time =", time.time() - t)
    print(output.shape)

if __name__ == "__main__":
    main()
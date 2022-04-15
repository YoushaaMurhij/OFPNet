# Author: Zylo117

import torch
from torch import nn

from core.models.efficientdet.model import BiFPN, EfficientNet


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }


        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])


        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        # self.up = nn.ConvTranspose2d(23, 23, kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][0], conv_channel_coef[compound_coef][0], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][1], conv_channel_coef[compound_coef][1], kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][2], conv_channel_coef[compound_coef][2], kernel_size=2, stride=2)

        self.up_final1 = nn.ConvTranspose2d(88, 88, kernel_size=2, stride=2)  
        
        self.conv_final1 = nn.Sequential(
            nn.Conv2d(88, 44, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(44),
            nn.ReLU(inplace=True),
        )

        self.up_final2 = nn.ConvTranspose2d(44, 44, kernel_size=2, stride=2)  
        
        self.conv_final2 = nn.Sequential(
            nn.Conv2d(44, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        
        # inputs = self.up(inputs)

        _, p3, p4, p5 = self.backbone_net(inputs)

        p3 = self.up3(p3)
        p4 = self.up4(p4)
        p5 = self.up5(p5)

        features = (p3, p4, p5)
        # print("features back")
        # for f in features:
        #     print(f.shape)

        features = self.bifpn(features)
        # print("features")
        # for f in features:
        #     print(f.shape)

        x = self.up_final1(features[0])
        x = self.conv_final1(x)
        x = self.up_final2(x)
        x = self.conv_final2(x)
        return x 

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

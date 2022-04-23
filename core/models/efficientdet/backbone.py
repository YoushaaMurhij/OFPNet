import torch
from torch import nn
from core.models.efficientdet.model import BiFPN, EfficientNet

class EfficientFlow(nn.Module):
    def __init__(self, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientFlow, self).__init__()
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

        self.up3 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][0], conv_channel_coef[compound_coef][0], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][1], conv_channel_coef[compound_coef][1], kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(conv_channel_coef[compound_coef][2], conv_channel_coef[compound_coef][2], kernel_size=2, stride=2)

        self.up_final1 = nn.ConvTranspose2d(self.fpn_num_filters[self.compound_coef], self.fpn_num_filters[self.compound_coef], kernel_size=2, stride=2)  
        
        self.conv_final1 = nn.Sequential(
            nn.Conv2d(self.fpn_num_filters[self.compound_coef], self.fpn_num_filters[self.compound_coef] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_num_filters[self.compound_coef] // 2),
            nn.ReLU(inplace=True),
        )

        self.up_final2 = nn.ConvTranspose2d(self.fpn_num_filters[self.compound_coef] // 2, self.fpn_num_filters[self.compound_coef] // 2, kernel_size=2, stride=2)  

        self.fc_extract = nn.Linear(23, 23)
        self.fc1 = nn.Linear(self.fpn_num_filters[self.compound_coef] // 2, 32)
        self.fc2 = nn.Linear(32, 32)
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = torch.permute(inputs, (0, 2, 3, 1))
        x = self.fc_extract(x)
        x = torch.permute(x, (0, 3, 1, 2))

        _, p3, p4, p5 = self.backbone_net(x)

        p3 = self.up3(p3)
        p4 = self.up4(p4)
        p5 = self.up5(p5)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        x = self.up_final1(features[0])
        x = self.conv_final1(x)
        x = self.up_final2(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.permute(x, (0, 3, 1, 2))

        return x 

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

def main():
    device =  'cuda:0'
    model = EfficientFlow(compound_coef=1).to(device)
    print(model)
    x = torch.rand((1, 23, 256, 256)).to(device)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    main()
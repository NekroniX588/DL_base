import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), expand_ratio=1):
        super(DeepWiseBlock, self).__init__()
        
        hidden_dim = round(in_channels * expand_ratio)
        self.identity = stride[0] == 1 and stride[1] == 1 and in_channels == out_channels
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
                
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class OCREncoder(nn.Module):
    def __init__(self):
        super(OCREncoder, self).__init__()
        
        self.init_conv = DeepWiseBlock(1, 16)
        self.layer1 = nn.Sequential(
            DeepWiseBlock(16, 32),
            DeepWiseBlock(32, 32, stride=(2, 2), expand_ratio=1),
        )
        self.layer2 = nn.Sequential(
            DeepWiseBlock(32, 64),
            DeepWiseBlock(64, 64, expand_ratio=2),
            DeepWiseBlock(64, 64, stride=(2, 2), expand_ratio=1),
        )
        self.layer3 = nn.Sequential(
            DeepWiseBlock(64, 128),
            DeepWiseBlock(128, 128, expand_ratio=2),
            DeepWiseBlock(128, 128, stride=(2, 2), expand_ratio=1),
        )
        self.layer4 = nn.Sequential(
            DeepWiseBlock(128, 128, expand_ratio=2),
            DeepWiseBlock(128, 128, expand_ratio=2),
            DeepWiseBlock(128, 128, stride=(2, 2), expand_ratio=1),
        )
        self.maxpool = nn.AdaptiveAvgPool2d((5, 1))
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.init_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.maxpool(x)
        
        x = x.squeeze(3)
        x = x.transpose(1, 2)
        return x
    
class OCRDecoder(nn.Module):
    def __init__(self, output_dim=19):
        super(OCRDecoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=128, 
                            num_layers=2, 
                            bias=True, 
                            batch_first=True, 
                            dropout=0.2, 
                            bidirectional=True
                           )
        self.output = nn.Linear(256, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(2 * 2, x.size(0), 128, device=x.device)
        c0 = torch.randn(2 * 2, x.size(0), 128, device=x.device)
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.output(output)
        return output
        
class OCRModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(OCRModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
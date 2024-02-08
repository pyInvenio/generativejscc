import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function

from util_module import AF_Module, AttentionDecoder, AttentionEncoder, GDN, PowerNormalization, ResidualBlockUpsample
from stylegan import StyleGan


    
class AWGN():
    def __init__(self, snr) -> None:
        self.snr = snr
        
    def work(self, input, output):
        symboles_in = input
        noise = np.sqrt(10 **(-self.snr/10))
        awgn_filter = noise*np.random.randn(len(symboles_in))
        
        for i, symbol in enumerate(symboles_in):
            output[i] = symbol + awgn_filter[i]
            
        return output


class ResidualBlockS2(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1):
        super(ResidualBlockS2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gdn = GDN(out_ch, device='cuda')
        self.downsample = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            )
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out += self.downsample(identity)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out += self.downsample(identity)
        return out
    
class F_Theta_Network_Encoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(F_Theta_Network_Encoder, self).__init__()

        self.layer1 = ResidualBlock(in_channels, 512, stride=2)
        self.af_module1 = AF_Module(512)
        self.layer2 = ResidualBlock(512, 512)
        self.layer3 = ResidualBlock(512, 512, stride=2)
        self.af_module2 = AF_Module(512)
        self.attention_module = AttentionEncoder(512)
        self.layer4 = ResidualBlock(512, 512)
        self.layer5 = ResidualBlock(512, 512, stride=2)
        self.af_module3 = AF_Module(512)
        self.layer6 = ResidualBlock(512, 512)
        self.layer7 = ResidualBlock(512, 512, stride=2)
        self.af_module4 = AF_Module(num_classes)  # Assuming num_classes is Cout
        self.attention_module2 = AttentionEncoder(num_classes)
        self.power_norm = PowerNormalization(num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.af_module1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.af_module2(out)
        out = self.attention_module(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.af_module3(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.af_module4(out)
        out = self.attention_module2(out)
        out = self.power_norm(out)
        return out

class G_Phi_Network_Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(G_Phi_Network_Decoder, self).__init__()
        self.attention_module = AttentionDecoder(num_classes)
        self.layer1 = ResidualBlock(512, 512)
        self.layer2 = ResidualBlockUpsample(512, 512)
        self.af_module1 = AF_Module(512)
        self.layer3 = ResidualBlock(512, 512)
        self.layer4 = ResidualBlockUpsample(512, 512)
        self.af_module2 = AF_Module(512)
        self.attention_module2 = AttentionDecoder(512)
        self.layer5 = ResidualBlock(512, 512)
        self.layer6 = ResidualBlockUpsample(512, 512)
        self.af_module3 = AF_Module(512)
        self.layer7 = ResidualBlock(512, 512)
        self.layer8 = ResidualBlockUpsample(512, 512)
        self.af_module4 = AF_Module(512)
        
    def forward(self, x):
        out = self.attention_module(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.af_module1(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.af_module2(out)
        out = self.attention_module2(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.af_module3(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.af_module4(out)
        return out
    
class ForwardOperator(nn.Module):
    
    def __init__(self):
        super(ForwardOperator, self).__init__()
        self.encoder = F_Theta_Network_Encoder(3, 512)
        self.decoder = G_Phi_Network_Decoder(512, 3)
        self.awgn = AWGN(10)
        self.loss = nn.MSELoss()
        self.gan = StyleGan('pretrained/ffhq-512-avg-tpurun1.pkl', 0.5, 'random', 'cuda')
        
    def forward(self, x):
        x = self.encoder(x)
        self.awgn.work(x, x)
        x_hat = self.decoder(x)
        
        ganx = self.gan.G(x_hat)
        gan_x_hat = self.encoder(ganx)
        self.awgn.work(gan_x_hat, gan_x_hat)
        gan_x_hat = self.decoder(gan_x_hat)
        
        mse_loss = self.loss(x_hat, gan_x_hat)
        return x_hat, gan_x_hat, mse_loss
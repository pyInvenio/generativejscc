import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function

from util_module import AF_Module, GDN, PowerNormalization, ResidualBlockUpsample, NonLocalAttentionBlock
from stylegan import StyleGan

num_filters = 512

def random_snr():
    return np.random.randint(-5, 5)
    
class AWGN():
    def __init__(self, snr):
        self.snr = snr
        
    def work(self, input, output):
        symboles_in = input.clone()  # Clone the input tensor to prevent in-place modification
        noise = torch.sqrt(torch.tensor(10 **(-self.snr/10), dtype=torch.float32))  # Convert noise to tensor
        awgn_filter = noise * torch.randn_like(symboles_in)  # Generate noise tensor
        
        output[:] = symboles_in + awgn_filter  # Add noise to input and assign to output


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
        self.af_module1 = AF_Module(512, 1)
        self.layer2 = ResidualBlock(512, 512)
        self.layer3 = ResidualBlock(512, 512, stride=2)
        self.af_module2 = AF_Module(512, 1)
        self.attention_module = NonLocalAttentionBlock(num_filters)
        self.layer4 = ResidualBlock(512, 512)
        self.layer5 = ResidualBlock(512, 512, stride=2)
        self.af_module3 = AF_Module(512, 1)
        self.layer6 = ResidualBlock(512, 512)
        self.layer7 = ResidualBlock(512, 512, stride=2)
        self.af_module4 = AF_Module(num_classes, 1)  # Assuming num_classes is Cout
        self.attention_module2 = NonLocalAttentionBlock(num_filters)
        self.power_norm = PowerNormalization((num_classes,))

    def forward(self, x):
        snr =  torch.randn(1).to('cuda')
        out = self.layer1(x)
        out = self.af_module1(out, snr)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.af_module2(out, snr)
        out = self.attention_module(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.af_module3(out, snr)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.af_module4(out, snr)
        out = self.attention_module2(out)
        out = self.power_norm(out)
        return out

class G_Phi_Network_Decoder(nn.Module):
    def __init__(self, in_channels, num_classes, device):
        super(G_Phi_Network_Decoder, self).__init__()
        self.attention_module = NonLocalAttentionBlock(num_filters)
        self.layer1 = ResidualBlock(512, 512)
        self.layer2 = ResidualBlockUpsample(512, 512, device)
        self.af_module1 = AF_Module(512, 1)
        self.layer3 = ResidualBlock(512, 512)
        self.layer4 = ResidualBlockUpsample(512, 512, device)
        self.af_module2 = AF_Module(512, 1)
        self.attention_module2 = NonLocalAttentionBlock(num_filters)
        self.layer5 = ResidualBlock(512, 512)
        self.layer6 = ResidualBlockUpsample(512, 512, device)
        self.af_module3 = AF_Module(512, 1)
        self.layer7 = ResidualBlock(512, 512)
        self.layer8 = ResidualBlockUpsample(512, 512, device)
        self.af_module4 = AF_Module(3, 1)
        
    def forward(self, x):
        snr =  torch.randn(1).to('cuda')
        out = self.attention_module(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.af_module1(out, snr)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.af_module2(out, snr)
        out = self.attention_module2(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.af_module3(out, snr)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.af_module4(out, snr)
        return out
    
class ForwardOperator(nn.Module):
    
    def __init__(self, device):
        super(ForwardOperator, self).__init__()
        self.encoder = F_Theta_Network_Encoder(3, 512)
        self.decoder = G_Phi_Network_Decoder(512, 3, device)
        self.awgn = AWGN(10)
        self.loss = nn.MSELoss()
        self.gan = StyleGan('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl', 0.5, 'random', 'cuda')
        
    def forward(self, x):
        # Assuming x is of size (batch_size, channels, height, width) where batch_size = 64
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)  # Reshape to match CIFAR10 image size
        
        x = self.encoder(x)
        self.awgn.work(x, x)
        x_hat = self.decoder(x)
        
        ganx = self.gan.G(x_hat)
        gan_x_hat = self.encoder(ganx)
        self.awgn.work(gan_x_hat, gan_x_hat)
        gan_x_hat = self.decoder(gan_x_hat)
        
        mse_loss = self.loss(x_hat, gan_x_hat)
        return x_hat, gan_x_hat, mse_loss

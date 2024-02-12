import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
    
    
class Concatenate(nn.Module):
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, tensors):
        return torch.cat(tensors, dim=self.dim)


class GFR_Encoder_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super(GFR_Encoder_Module, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same')
        self.gdn = GDN(out_channels)
        self.activation = None
        if activation == 'prelu':
            self.activation = nn.PReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.gdn(out)
        if self.activation:
            out = self.activation(out)
        return out

class GFR_Decoder_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super(GFR_Decoder_Module, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same')
        self.gdn = GDN(out_channels)
        self.activation = None
        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.gdn(out)
        if self.activation:
            out = self.activation(out)
        return out
    
class AF_Module(nn.Module):
    def __init__(self, channels, snr_size):
        super(AF_Module, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.concat = Concatenate()
        self.dense1 = nn.Linear(channels + snr_size, channels // 16)
        self.dense2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr):
        m = self.global_pooling(x)
        m = self.concat([m, snr])
        m = F.relu(self.dense1(m))
        m = self.sigmoid(self.dense2(m))
        out = x * m
        return out


class AttentionEncoder(nn.Module):
    def __init__(self, in_channels, snr_size, tcn):
        super(AttentionEncoder, self).__init__()
        self.en1 = GFR_Encoder_Module(in_channels, 256, (9, 9), 2, 'prelu')
        self.en1_af = AF_Module(256, snr_size)
        self.en2 = GFR_Encoder_Module(256, 256, (5, 5), 2, 'prelu')
        self.en2_af = AF_Module(256, snr_size)
        self.en3 = GFR_Encoder_Module(256, 256, (5, 5), 1, 'prelu')
        self.en3_af = AF_Module(256, snr_size)
        self.en4 = GFR_Encoder_Module(256, 256, (5, 5), 1, 'prelu')
        self.en4_af = AF_Module(256, snr_size)
        self.en5 = GFR_Encoder_Module(256, tcn, (5, 5), 1)

    def forward(self, x, snr):
        x = self.en1(x)
        x = self.en1_af(x, snr)
        x = self.en2(x)
        x = self.en2_af(x, snr)
        x = self.en3(x)
        x = self.en3_af(x, snr)
        x = self.en4(x)
        x = self.en4_af(x, snr)
        x = self.en5(x)
        return x
class AttentionDecoder(nn.Module):
    def __init__(self, in_channels, snr_size, tcn):
        super(AttentionDecoder, self).__init__()
        self.de1 = GFR_Decoder_Module(256, 256, (5, 5), 1, 'prelu')
        self.de1_af = AF_Module(256, snr_size)
        self.de2 = GFR_Decoder_Module(256, 256, (5, 5), 1, 'prelu')
        self.de2_af = AF_Module(256, snr_size)
        self.de3 = GFR_Decoder_Module(256, 256, (5, 5), 1, 'prelu')
        self.de3_af = AF_Module(256, snr_size)
        self.de4 = GFR_Decoder_Module(256, 256, (5, 5), 2, 'prelu')
        self.de4_af = AF_Module(256, snr_size)
        self.de5 = GFR_Decoder_Module(256, 3, (9, 9), 2, 'sigmoid')

    def forward(self, x, snr):
        x = self.de1(x)
        x = self.de1_af(x, snr)
        x = self.de2(x)
        x = self.de2_af(x, snr)
        x = self.de3(x)
        x = self.de3_af(x, snr)
        x = self.de4(x)
        x = self.de4_af(x, snr)
        x = self.de5(x)
        return x

class PowerNormalization(nn.Module):
    def __init__(self, codeword_shape):
        super(PowerNormalization, self).__init__()
        self.codeword_shape = codeword_shape

    def forward(self, codeword):
        codeword = codeword.view(1, -1)
        n_vals = codeword.size(1)
        normalized = (codeword / torch.norm(codeword, dim=1, keepdim=True)) * torch.sqrt(torch.tensor(n_vals).float())
        ch_input = normalized.view(self.codeword_shape)
        return ch_input
    
    
class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(ResidualBlockUpsample, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 3 * out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)

        self.conv2 = nn.Conv2d(4 * out_channels, 3 * out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(3 * out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gdn = GDN(out_channels, device=device)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
    
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.pixel_shuffle1(out)
        
        out = torch.cat([out, identity], dim=1)
        out = self.conv2(out)
        out = self.pixel_shuffle2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.gdn(out)
        
        out += identity
        return out
    
    

class SignalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, corr=True, strides_down=1, padding="same"):
        super(SignalConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides_down, padding=padding)
        self.activation = nn.ReLU()  # ReLU activation function
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv0 = SignalConv2D(num_filters//2, num_filters//2, kernel_size=(1, 1))
        self.conv1 = SignalConv2D(num_filters//2, num_filters//2, kernel_size=(3, 3))
        self.conv2 = SignalConv2D(num_filters//2, num_filters, kernel_size=(1, 1))
        
    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out += residual
        return out

class NonLocalAttentionBlock(nn.Module):
    def __init__(self, num_filters):
        super(NonLocalAttentionBlock, self).__init__()
        self.trunk_branch = nn.Sequential(
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters)
        )
        self.attention_branch = nn.Sequential(
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters)
        )
        self.conv1x1 = SignalConv2D(num_filters, num_filters, kernel_size=(1, 1))
        
    def forward(self, x):
        trunk_branch_out = self.trunk_branch(x)
        attention_branch_out = self.attention_branch(x)
        attention_branch_out = self.conv1x1(attention_branch_out)
        attention_branch_out = torch.sigmoid(attention_branch_out)
        out = x + attention_branch_out * trunk_branch_out
        return out
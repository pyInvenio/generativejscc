import numpy as np
import torch.nn as nn
import torch

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


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1):
        super(ResidualBlock, self).__init__()
        
        
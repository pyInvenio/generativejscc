import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from tqdm import tqdm
import torch.nn.functional as F

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading

import cv2

class genjscc():
    
    def __init__(self, model_cout, image_path):
        self.cfx = cuda.Device(0).make_context()
        self.target_dtype = np.float32
        self.model_cout = model_cout
        self.image_path = image_path
        
    def _get_image(self, image_path):
        image = cv2.imread(image_path)
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 0, 2)
        image = np.expand_dims(image, axis=0)/255.0
        image = np.ascontiguousarray(image, dtype=self.target_dtype)
        self.image_shape = image.shape
        self.codeword_shape = (1, self.model_cout, self.image_shape[2]//16, self.image_shape[3]//16)
        return image
    
    def _power_normalize(self, codeword):
        codeword = codeword.reshape(1, -1)
        n_vales = codeword.shape[1]
        normalized = (codeword/np.linalg.norm(codeword, axis=1, keepdims=True))*np.sqrt(n_vales)
        ch_input = normalized.reshape(self.codeword_shape)
        return ch_input
    
    def _image_encode(self, image, snr):
        threading.Thread.__init__(self)
        self.cfx.push()
        
        output = np.empty(self.codeword_shape, dtype=self.target_dtype)
        
        ch_codeword = self._power_normalize(image)
    
    def _pre_encode(self):
        print("Pre-encoding...")
        codeword = self._image_encode(self._get_image(self.image_path), self.snr)
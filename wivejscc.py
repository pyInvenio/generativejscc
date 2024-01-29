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
    
    def __init__(self, model_cout, key_encoder_fn, interp_encoder_fn, image_path, snr, num_chunks, packet_len):
        self.cfx = cuda.Device(0).make_context()
        self.target_dtype = np.float32
        self.model_cout = model_cout
        self.image_path = image_path
        self.packet_len = packet_len
        self.snr = np.array([[snr]], dtype=self.target_dtype)
        self._get_runtime(trt.Logger(trt.Logger.WARNING))
        self.key_encoder = self._get_context(key_encoder_fn)
        self.interp_encoder = self._get_context(interp_encoder_fn)
        self.num_chunks = num_chunks
        self.chunk_size = model_cout // num_chunks
        
        self.ssf_sigma = 0.01
        self.ssf_levels = 5
        
        
    def _allocate_memory(self):
        self.codeword_addr = cuda.mem_alloc(np.empty(self.codeword_shape, dtype=self.target_dtype).nbytes)
        self.image_addr = cuda.mem_alloc(np.empty(self.image_shape, dtype=self.target_dtype).nbytes)
        self.interp_input_addr = cuda.mem_alloc(np.empty(self.interp_input_shape, dtype=self.target_dtype).nbytes)
        self.snr_addr = cuda.mem_alloc(self.snr.nbytes)
        self.ssf_input_addr = cuda.mem_alloc(np.empty(self.ssf_input_shape, dtype=self.target_dtype).nbytes)
        self.ssf_est_addr = cuda.mem_alloc(np.empty(self.image_shape, dtype=self.target_dtype).nbytes)
        
    def _get_image(self, image_path):
        image = cv2.imread(image_path)
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 0, 2)
        image = np.expand_dims(image, axis=0)/255.0
        image = np.ascontiguousarray(image, dtype=self.target_dtype)
        self.image_shape = image.shape
        self.codeword_shape = (1, self.model_cout, self.image_shape[2]//16, self.image_shape[3]//16)
        self.interp_input_shape = (1, 21, self.image_shape[2], self.image_shape[3])
        self.ssf_input_shape = (1, 6, self.image_shape[2], self.image_shape[3])
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
        
        bindings = [int(self.image_addr), int(self.snr_addr), int(self.code_addr)]
        
        cuda.memcpy_htod_async(self.image_addr, image, self.stream)
        cuda.memcpy_htod_async(self.snr_addr, snr, self.stream)
        self.key_encoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, self.codeword_addr, self.stream)
        self.stream.synchronize()
        
        ch_codeword = self._power_normalize(image)
        
        self.cfx.pop()
        
        return ch_codeword
    
    def _pre_encode(self):
        self._pre_encode_codewords = []
        print("Pre-encoding...")
        codeword = self._image_encode(self._get_image(self.image_path), self.snr)
        codeword = codeword.reshape(-1, 2)
        codeword = np.concatenate((codeword.reshape(-1, 2), np.zeros((self.n_padding, 2))), axis=0)
        codeword *= 0.1
        
    def work(self, input_items, output_items):
        payload_out = output_items[0]
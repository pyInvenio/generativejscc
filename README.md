Repository to reproduce results of https://arxiv.org/abs/2211.13772

Attention Module repo: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py

StyleGAN2 Faces (FFHQ 512x512) pretrained: https://mega.nz/file/eQdHkShY#8wyNKs343L7YUjwXlEg3cWjqK2g2EAIdYz5xbkPy3ng

pip install ninja

make sure ```nvcc test_nvcc.cu -o test_nvcc -run``` works
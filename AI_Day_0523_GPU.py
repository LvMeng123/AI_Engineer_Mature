#MLP框架练习
import torch 
import numpy as np

class GPUPractice:
    
    def __init__(self):
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            x_cuda = torch.randn(3,3).cuda()
            print(x_cuda)
    

Gpu = GPUPractice()

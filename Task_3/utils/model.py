import torch
import mytensor

import numpy as np

class Model():
    def __init__(self,*args):
        self.layers = []
        for layer in args:
            self.layers.append(layer)

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x) 
 
        return x
    
    def backward(self,x):
        for layer in reversed(self.layers):
            x = layer.backward(x)

        return x
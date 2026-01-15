import torch
import mytensor
import numpy as np

from .model import Model

class Optimizer():
    def __init__(self,model,lr):
        self.model = model
        self.lr = lr
        
    
    def step(self):
        for layer in self.model.layers:
            if isinstance(layer,mytensor.Linear):
                layer.update_weights(self.lr)
            elif isinstance(layer,mytensor.Conv):
                layer.update_weights(self.lr)
            else:
                continue
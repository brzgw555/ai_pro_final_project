import torch
import mytensor
import numpy as np

from .optimizer import Optimizer
from .model import Model

class Scheduler():
    def __init__(self,optimizer, start_lr):
        self.optimizer = optimizer
        self.start_lr = start_lr
    
        

class StepLR(Scheduler):
    def __init__(self,optimizer, start_lr, step_size, gamma):
        super().__init__(optimizer, start_lr)
        self.step_size = step_size
        self.gamma = gamma
        
        self.cnt= 0
        
        

    def step_ratio(self):
        self.optimizer.lr*= self.gamma

    def step(self):
        self.cnt+=1
        if self.cnt%self.step_size==0:
            self.step_ratio()


class CosineAnnealingLR(Scheduler):
    def __init__(self,optimizer, start_lr, T_max, eta_min):
        super().__init__(optimizer, start_lr)
        self.T_max = T_max
        self.eta_min = eta_min
        
        self.cnt = 0
        
    def step_cos(self):
        
        self.optimizer.lr =  self.eta_min + (self.start_lr - self.eta_min) * (1 + np.cos(np.pi * self.cnt / self.T_max)) / 2

    def step(self):
        self.cnt+=1
        
        self.step_cos()

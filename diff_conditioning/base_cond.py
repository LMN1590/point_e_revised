import torch

class BaseCond:
    def __init__(self, grad_scale:float, *args):
        self.grad_scale = grad_scale
        self.loss_lst = []
        self.grad_lst = []
    
    def calculate_gradient(self, x:torch.Tensor,t:torch.Tensor, **model_kwargs):
        loss = self.calculate_loss(x,t,**model_kwargs)
        self.loss_lst.append(loss)
        
        grad = torch.autograd.grad(loss, x)[0]  # gradient wrt x
        self.grad_lst.append(grad)
        return -grad*self.grad_scale  # negative sign: push mean back toward origin
        
    
    def calculate_loss(self, x:torch.Tensor,t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        raise NotImplementedError("Base Loss Calculation called")
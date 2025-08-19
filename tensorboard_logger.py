#!/usr/bin/env python3
"""
TensorBoard Setup for Real-time Training Monitoring
Complete example showing how to log metrics, gradients, weights, and images to TensorBoard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

class TrainingLogger:
    """Enhanced TensorBoard logger for training metrics"""
    
    def __init__(self, log_dir=None, experiment_name=None):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = experiment_name or "experiment"
            log_dir = f"runs/{experiment_name}_{timestamp}"
        
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        
        print(f"📊 TensorBoard logging to: {log_dir}")
        print(f"🌐 Run: tensorboard --logdir={log_dir} --port=6006")
        print(f"🔗 Open: http://localhost:6006")
    
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value"""
        step = step or self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, values_dict, step=None):
        """Log multiple scalars together"""
        step = step or self.step
        self.writer.add_scalars(tag, values_dict, step)
    
    def log_histogram(self, tag, values, step=None):
        """Log histogram of values (weights, gradients, etc.)"""
        step = step or self.step
        if torch.is_tensor(values):
            values = values.detach().cpu()
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, image, step=None):
        """Log an image"""
        step = step or self.step
        self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure"""
        step = step or self.step
        self.writer.add_figure(tag, figure, step)
    
    def log_model_weights(self, model, step=None):
        """Log all model weights as histograms"""
        step = step or self.step
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param.detach().cpu(), step)
    
    def log_model_gradients(self, model, step=None):
        """Log all model gradients as histograms"""
        step = step or self.step
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad.detach().cpu(), step)
    
    def log_gradient_norms(self, model, step=None):
        """Log gradient norms per layer"""
        step = step or self.step
        grad_norms = {}
        total_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        grad_norms['total'] = total_norm
        
        for name, norm in grad_norms.items():
            self.writer.add_scalar(f'gradient_norms/{name}', norm, step)
        
        return total_norm
    
    def log_learning_rate(self, optimizer, step=None):
        """Log current learning rate"""
        step = step or self.step
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
    
    def increment_step(self):
        """Increment the global step counter"""
        self.step += 1
    
    def close(self):
        """Close the writer"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Quick setup for your existing training code
def quick_tensorboard_setup(experiment_name="training"):
    """Quick setup function for existing training code"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{experiment_name}_{timestamp}"
    writer = TrainingLogger(log_dir,experiment_name)
    
    print(f"📊 TensorBoard logging to: {log_dir}")
    print(f"🌐 To view, run in terminal:")
    print(f"    tensorboard --logdir=runs --port=6006")
    print(f"🔗 Then open: http://localhost:6006")
    
    return writer

# Integration template for your existing code
"""
# Add this to your training script:

# At the top of your script
from torch.utils.tensorboard import SummaryWriter
writer = quick_tensorboard_setup("my_training")

# In your training loop
step = 0
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... your existing training code ...
        
        loss.backward()
        
        # Log metrics
        writer.add_scalar('Loss/Train', loss.item(), step)
        
        # Log gradients (every N steps to avoid slowdown)
        if step % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    writer.add_scalar(f'GradientNorms/{name}', param.grad.norm().item(), step)
        
        optimizer.step()
        step += 1

# Close writer when done
writer.close()
"""

tensorboard_logger = quick_tensorboard_setup("hand_grad_ddim256_scale1e0_k3_thresh96_updated_grippingloss")
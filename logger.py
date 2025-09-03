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
import logging
import csv

class TrainingLogger:
    """Enhanced TensorBoard logger for training metrics"""
    
    def __init__(self, log_dir=None, experiment_name=None,increment_step:float=1.):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = experiment_name or "experiment"
            log_dir = f"runs/{experiment_name}_{timestamp}"
        
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.increment_step = increment_step
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
    
    def increment(self):
        """Increment the global step counter"""
        self.step += self.increment_step
    
    def close(self):
        """Close the writer"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        # Always overwrite file and write header
        self.file = open(filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row_dict):
        # Check for unexpected keys
        extra_keys = set(row_dict.keys()) - set(self.fieldnames)
        if extra_keys:
            raise KeyError(f"Unexpected keys in log: {extra_keys}")

        # Fill missing keys with None
        safe_row = {k: row_dict.get(k, None) for k in self.fieldnames}
        self.writer.writerow(safe_row)
        self.file.flush()

    def close(self):
        self.file.close()


# Quick setup for your existing training code
def quick_tensorboard_setup(experiment_name:str,log_dir:str,increment_step:float):
    """Quick setup function for existing training code"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{experiment_name}_{timestamp}"
    writer = TrainingLogger(log_dir,experiment_name,increment_step=increment_step)
    
    return writer

def init_all_logger(out_dir:str, exp_name:str, tensorboard_log_dir:str,increment_step:float):
    """Initialize TensorBoard logger"""
    global TENSORBOARD_LOGGER,CSVLOGGER
    TENSORBOARD_LOGGER = quick_tensorboard_setup(exp_name,log_dir=tensorboard_log_dir,increment_step=increment_step)
    CSVLOGGER = CSVLogger(
        filepath = os.path.join(out_dir,exp_name,'training_log.csv'),
        fieldnames = [
            'phase',
            'sampling_step','local_iter','batch_idx',
            "sap_epoch", 'sap_loss','sap_inputs_grad_norm','sap_lr','sap_num_points',
            'softzoo_loss','softzoo_grad_norm','softzoo_reward',
            'softzoo_mean_loss','softzoo_scaled_mean_grad_norm',
            
            'loss','grad_norm','note'
        ]
    )
    logging.basicConfig(
        filename = os.path.join(out_dir,exp_name,'training.log'),
        filemode='a',
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
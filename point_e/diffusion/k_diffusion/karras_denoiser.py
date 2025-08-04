"""
Based on: https://github.com/crowsonkb/k-diffusion

Copyright (c) 2022 Katherine Crowson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import torch as th
import torch.nn as nn

from ..diff_utils import mean_flat
from .karras_utils import append_dims,append_zero

class KarrasDenoiser:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def get_snr(self, sigmas:th.Tensor):
        return sigmas**-2

    def get_sigmas(self, sigmas:th.Tensor):
        return sigmas

    def get_scalings(self, sigma:th.Tensor):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model:nn.Module, x_start:th.Tensor, sigmas:th.Tensor, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        c_skip, c_out, _ = [append_dims(x, dims) for x in self.get_scalings(sigmas)]
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        target = (x_start - c_skip * x_t) / c_out

        terms["mse"] = mean_flat((model_output - target) ** 2)
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def denoise(self, model:nn.Module, x_t:th.Tensor, sigmas:th.Tensor, **model_kwargs):
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised
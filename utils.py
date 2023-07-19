import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import LinearOperator, eigsh

class DCTLowPasser(nn.Module):
    def __init__(self, module, bandwidth):
        super().__init__()
        self.module = module
        self.apply_filter = True
        self.bandwidth = bandwidth

    def forward(self, x):
        if self.apply_filter:
            x = dct_low_pass(x, self.bandwidth)
        return self.module(x)


def dct_low_pass(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.zeros_like(x)
    mask[:, :bandwidth, :bandwidth] = 1
    return idct_2d(dct_2d(x, norm="ortho") * mask, norm="ortho").squeeze_()


def dct_high_pass(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.zeros_like(x)
    mask[:, -bandwidth:, -bandwidth:] = 1
    return idct_2d(dct_2d(x, norm="ortho") * mask, norm="ortho").squeeze_()


def dct_cutoff_low(x, bandwidth):
    if len(x.size()) == 2:
        x.unsqueeze_(0)

    mask = torch.ones_like(x)
    mask[:, :bandwidth, :bandwidth] = 0
    return idct_2d(dct_2d(x, norm="ortho") * mask, norm="ortho").squeeze_()


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = V_r + V_i * 1.0j

    v = torch.fft.ifft(V).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


def hessian_vector_product(model, x, y, v, h):
    z_1 = x + h * v
    z_1.requires_grad = True
    output = model(z_1)
    loss_z_1 = F.cross_entropy(output, y, reduction="sum")
    loss_z_1.backward()
    grad_z_1 = z_1.grad.detach()

    z_2 = x - h * v
    z_2.requires_grad = True
    output = model(z_2)
    loss_z_2 = F.cross_entropy(output, y, reduction="sum")
    loss_z_2.backward()
    grad_z_2 = z_2.grad.detach()

    return (grad_z_1 - grad_z_2) / 2 * h


def evd_hessian(model, x, y, h, k):
    def _mv(v):
        v_ = torch.tensor(v, device=x.device, dtype=torch.float32).view(x.shape)
        return hessian_vector_product(model, x, y, v_, h).view(-1).cpu()

    A = LinearOperator((np.prod(x.shape), np.prod(x.shape)), matvec=_mv)
    eig_values, eig_vecs = eigsh(A, k, which="LA")
    return eig_values, eig_vecs

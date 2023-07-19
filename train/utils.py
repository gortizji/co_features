import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import l2_norm_batch


class Trainer:

    def train(self, train_loader, test_loader=None):
        raise NotImplementedError

    def evaluate(self, test_loader=None):
        raise NotImplementedError

    def log(self, log_dict, explicit_print=False):
        print(log_dict)


class GradAlign(nn.Module):
    def __init__(self, reg_lambda, epsilon) -> None:
        super().__init__()
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon

    def forward(self, model, loss, delta, X, y):
        reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly

        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
        grad = grad.detach()

        if self.reg_lambda > 0:
            grad2 = self.__get_input_grad(model, X, y)
            grads_nnz_idx = ((grad ** 2).sum([1, 2, 3]) ** 0.5 != 0) * (
                (grad2 ** 2).sum([1, 2, 3]) ** 0.5 != 0
            )
            grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
            grad1_norms, grad2_norms = l2_norm_batch(grad1), l2_norm_batch(grad2)
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
            reg += self.reg_lambda * (1.0 - cos.mean())

        return reg

    def __get_input_grad(self, model, X, y):
        delta = torch.zeros_like(X)
        delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
        return grad


def accuracy(output, y):
    return (output.max(1)[1] == y).float().mean()


def is_correct(output, y):
    return output.max(1)[1] == y


class Normalizer(nn.Module):
    def __init__(self, module, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.module = module

    def forward(self, x):
        input = (x - self.mean) / self.std
        return self.module(input)

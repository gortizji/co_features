import torch
import torch.nn as nn
import torch.nn.functional as F
from autoattack import AutoAttack

from train.utils import accuracy

from .utils import clip_box, l2_clip, l2_project


class Attack:
    def attack(self, model, X, y, warmup_scale=1.0):
        raise NotImplementedError


class FGSM(Attack):
    def __init__(self, epsilon, alpha, clip_delta=True, clip_box=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.clip_delta = clip_delta
        self.clip_box = clip_box
        super().__init__()

    def attack(self, model, X, y, warmup_scale=1.0):
        delta = self.init_delta(X, warmup_scale)
        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        # Bibi's changes >> NFGSM does not have a clipping operation only so that it is [0,1] of final image
        delta.data = delta + warmup_scale * self.alpha * torch.sign(grad)
        if self.clip_delta:
            delta.data = torch.clamp(
                delta, -self.epsilon * warmup_scale, self.epsilon * warmup_scale
            )
        if self.clip_box:
            delta.data = clip_box(delta, X)

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        return loss, delta, output

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X)
        delta.requires_grad = True
        return delta


class FGM(Attack):
    def __init__(self, epsilon, alpha, clip_delta=True, clip_box=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.clip_delta = clip_delta
        self.clip_box = clip_box
        super().__init__()

    def attack(self, model, X, y, warmup_scale=1.0):
        delta = self.init_delta(X, warmup_scale)
        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()

        # Bibi's changes >> NFGSM does not have a clipping operation only so that it is [0,1] of final image
        delta.data = delta + l2_project(grad, warmup_scale * self.alpha)
        if self.clip_delta:
            delta.data = l2_clip(delta, warmup_scale * self.epsilon)
        if self.clip_box:
            delta.data = clip_box(delta, X)

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        return loss, delta, output

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X)
        delta.requires_grad = True
        return delta


class RSFGSM(FGSM):
    def __init__(self, epsilon, alpha, unif, clip_delta=True, clip_box=True):
        self.unif = unif
        super().__init__(epsilon, alpha, clip_delta, clip_box)

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X)
        delta.uniform_(
            -self.unif * self.epsilon * warmup_scale, self.unif * self.epsilon * warmup_scale
        )
        if self.clip_box:
            delta = clip_box(delta, X)
        delta.requires_grad = True

        return delta


class NFGSM(RSFGSM):
    def __init__(self, epsilon, alpha, unif, clip_box=True):
        super().__init__(
            epsilon=epsilon, alpha=alpha, unif=unif, clip_delta=False, clip_box=clip_box
        )


class RSFGM(FGM):
    def __init__(self, epsilon, alpha, unif, clip_delta=True):
        self.unif = unif
        super().__init__(epsilon, alpha, clip_delta)

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X)
        delta.uniform_(
            -self.unif * self.epsilon * warmup_scale, self.unif * self.epsilon * warmup_scale
        )
        delta = l2_clip(delta, self.epsilon * self.unif * warmup_scale)
        delta = clip_box(delta, X)
        delta.requires_grad = True

        return delta


class LinfPGD(Attack):
    def __init__(self, epsilon, alpha, iterations, clip_box=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.clip_box = clip_box
        super().__init__()

    def attack(self, model, X, y, warmup_scale=1.0):

        delta = self.init_delta(X, warmup_scale)

        delta.requires_grad = True

        for _ in range(self.iterations):
            output = model(X + delta)

            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = torch.clamp(
                delta + warmup_scale * self.alpha * torch.sign(grad),
                -self.epsilon * warmup_scale,
                self.epsilon * warmup_scale,
            )
            if self.clip_box:
                delta.data = clip_box(delta, X)

            delta.grad.zero_()

        output = model(X + delta)
        loss = F.cross_entropy(output, y)

        return loss, delta, output

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X)
        delta.uniform_(-self.epsilon, self.epsilon)

        delta = clip_box(delta, X)

        return delta


class RestartPGD(LinfPGD):
    def __init__(self, epsilon, alpha, iterations, restarts, clip_box=True):
        super().__init__(epsilon, alpha, iterations, clip_box=clip_box)
        self.restarts = restarts

    def attack(self, model, X, y):

        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        max_output = torch.zeros_like(model(X)).cuda()
        for _ in range(self.restarts):
            delta = torch.zeros_like(X).cuda()

            _, delta, output = super().attack(model, X, y)

            all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_output[all_loss >= max_loss] = output.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_loss.mean(), max_delta, max_output


class L2PGD(Attack):
    def __init__(self, epsilon, alpha, iterations, clip_box=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.clip_box = clip_box
        super().__init__()

    def attack(self, model, X, y, warmup_scale=1.0):

        delta = self.init_delta(X, warmup_scale)

        delta.requires_grad = True

        for _ in range(self.iterations):
            output = model(X + delta)

            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()

            delta.data = delta + l2_project(grad, self.alpha * warmup_scale)
            delta.data = l2_clip(delta, self.epsilon * warmup_scale)

            if self.clip_box:
                delta.data = clip_box(delta, X)
            delta.grad.zero_()

        output = model(X + delta)
        loss = F.cross_entropy(output, y)

        return loss, delta, output

    def init_delta(self, X, warmup_scale=1.0):
        delta = torch.zeros_like(X).cuda()
        delta.normal_()

        delta = l2_clip(delta, self.epsilon * warmup_scale)
        delta = clip_box(delta, X)

        return delta


class RestartL2PGD(L2PGD):
    def __init__(self, epsilon, alpha, iterations, restarts, clip_box=True):
        super().__init__(epsilon, alpha, iterations, clip_box=clip_box)
        self.restarts = restarts

    def attack(self, model, X, y):

        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        max_output = torch.zeros_like(model(X)).cuda()
        for _ in range(self.restarts):
            delta = torch.zeros_like(X).cuda()

            _, delta, output = super().attack(model, X, y)

            all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_output[all_loss >= max_loss] = output.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return max_loss.mean(), max_delta, max_output


def evaluate_autoattack(test_loader, model, epsilon, batch_size):
    model.eval()

    autoattacker = AutoAttack(model, norm="Linf", eps=epsilon / 255)

    X = [x for (x, y) in test_loader]
    x_test = torch.cat(X, 0)
    Y = [y for (x, y) in test_loader]
    y_test = torch.cat(Y, 0)

    x_adv, y_adv = autoattacker.run_standard_evaluation(
        x_test, y_test, return_labels=True, bs=batch_size
    )
    adv_acc = (y_test == y_adv).float().mean().item()
    return adv_acc


def evaluate_attack(test_loader, model, attacker, log_prefix="test"):
    model.eval()

    test_acc = 0.0
    test_adv_acc = 0.0
    test_adv_loss = 0.0
    test_loss = 0.0
    for batch_it, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        adv_loss, _, adv_output = attacker.attack(model, X, y)

        test_adv_loss += adv_loss.mean().item()
        test_adv_acc += accuracy(adv_output, y).item()

        with torch.no_grad():
            output = model(X)
            test_loss += F.cross_entropy(output, y).item()
            test_acc += accuracy(output, y).item()

    return {
        f"{log_prefix}.adv_acc": test_adv_acc / (batch_it + 1),
        f"{log_prefix}.adv_loss": test_adv_loss / (batch_it + 1),
        f"{log_prefix}.acc": test_acc / (batch_it + 1),
        f"{log_prefix}.loss": test_loss / (batch_it + 1),
    }

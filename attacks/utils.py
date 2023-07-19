import torch


def l2_clip(delta, epsilon):
    norm = delta.view([delta.shape[0], -1]).norm(p=2, dim=-1)
    norm = torch.max(torch.full_like(norm, 1e-12), norm)
    if len(delta.shape) == 2:
        norm = norm[:, None]
    elif len(delta.shape) == 3:
        norm = norm[:, None, None]
    elif len(delta.shape) == 4:
        norm = norm[:, None, None, None]
    else:
        raise ValueError("Only inputs with 1,2 or 3 tensor dimensions are allowed.")
    return delta * torch.min(torch.ones_like(norm), epsilon / norm)


def l2_project(delta, epsilon):
    norm = delta.view([delta.shape[0], -1]).norm(p=2, dim=-1)
    norm = torch.max(torch.full_like(norm, 1e-12), norm)
    if len(delta.shape) == 2:
        norm = norm[:, None]
    elif len(delta.shape) == 3:
        norm = norm[:, None, None]
    elif len(delta.shape) == 4:
        norm = norm[:, None, None, None]
    else:
        raise ValueError("Only inputs with 1,2 or 3 tensor dimensions are allowed.")
    return delta * epsilon / norm


def clip_box(delta, X):
    delta.data = torch.clamp(delta, -X, 1.0 - X)
    return delta.data

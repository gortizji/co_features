import os

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from omegaconf import DictConfig, OmegaConf

from attacks import LinfPGD, RestartPGD, evaluate_attack
from data.injection import create_linf_carrier, inject_feature
from models.preact_resnet import PreActResNet18
from train.adv_train import AdvTrainer
from train.utils import Normalizer

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

@hydra.main(config_path="config/injected_feature_cifar_pgd_train", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Setting reproducibility stuff
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    trainset = torchvision.datasets.CIFAR10(
        hydra.utils.to_absolute_path(cfg.data_dir), train=True, transform=None, download=True
    )
    testset = torchvision.datasets.CIFAR10(
        hydra.utils.to_absolute_path(cfg.data_dir), train=False, transform=None, download=True
    )

    V = create_linf_carrier()

    train_loader = inject_feature(
        trainset,
        cfg.beta / 255,
        V=V,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    test_loader = inject_feature(
        testset,
        cfg.beta / 255,
        V=V,
        batch_size=cfg.batch_size_test,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    model = PreActResNet18().cuda()
    shifted_model = Normalizer(model, mean=mu, std=std)

    opt = torch.optim.SGD(
        shifted_model.parameters(),
        lr=cfg.lr_max,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    lr_steps = cfg.epochs * len(train_loader)

    if cfg.lr_schedule == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=cfg.lr_min,
            max_lr=cfg.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif cfg.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1
        )
    else:
        raise ValueError("Incorrect scheduler type")

    epsilon = cfg.epsilon / 255
    alpha = cfg.alpha / 255
    pgd_alpha = cfg.pgd_alpha / 255
    attack = LinfPGD(epsilon, alpha, iterations=cfg.iterations, clip_box=cfg.clip_box)
    eval_attack = RestartPGD(
        epsilon, pgd_alpha, cfg.pgd_iterations, restarts=cfg.pgd_restarts, clip_box=cfg.clip_box
    )

    trainer = AdvTrainer(
        model=shifted_model,
        epochs=cfg.epochs,
        scheduler=scheduler,
        optimizer=opt,
        attacker=attack,
        save_deltas=False,
    )
    trainer.train(train_loader, test_loader)
    eval_dict = evaluate_attack(test_loader, trainer.model, eval_attack, log_prefix="eval")
    print(eval_dict)



if __name__ == "__main__":
    main()

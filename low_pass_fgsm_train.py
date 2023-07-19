import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from attacks import FGSM, RestartPGD, evaluate_attack
from models.preact_resnet import PreActResNet18
from utils import DCTLowPasser
from data.cifar10 import get_loaders, mu, std
from train.adv_train import AdvTrainer
from train.utils import Normalizer

@hydra.main(config_path="config/low_pass_fgsm_train", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Setting reproducibility stuff
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    train_loader, test_loader = get_loaders(
        hydra.utils.to_absolute_path(cfg.data_dir),
        cfg.batch_size,
        batch_size_test=cfg.batch_size_test,
        num_workers=cfg.num_workers,
    )
    model = PreActResNet18().cuda()

    filtered_model = DCTLowPasser(module=model, bandwidth=cfg.bandwidth)
    filtered_model = Normalizer(filtered_model, mean=mu, std=std)

    opt = torch.optim.SGD(
        filtered_model.parameters(),
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

    epsilon = cfg.epsilon / 255.0
    alpha = epsilon
    pgd_alpha = cfg.pgd_alpha / 255.0
    attack = FGSM(epsilon, alpha, clip_delta=cfg.clip_delta, clip_box=cfg.clip_box)

    trainer = AdvTrainer(
        model=filtered_model, epochs=cfg.epochs, scheduler=scheduler, optimizer=opt, attacker=attack
    )
    trainer.train(train_loader, test_loader)

    eval_attack = RestartPGD(epsilon, pgd_alpha, cfg.pgd_iterations, cfg.pgd_restarts)
    eval_log_dict = evaluate_attack(test_loader, trainer.model, eval_attack, log_prefix="eval")
    wandb.log(eval_log_dict)

    trainer.model.apply_filter = False
    eval_attack = RestartPGD(epsilon, pgd_alpha, cfg.pgd_iterations, cfg.pgd_restarts)
    eval_log_dict = evaluate_attack(
        test_loader, trainer.model, eval_attack, log_prefix="eval.no_filter"
    )
    print(eval_log_dict)


if __name__ == "__main__":
    main()

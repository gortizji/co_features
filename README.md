# Catastrophic overfitting can be induced with discriminative non-robust features

This is the source code to reproduce the experiments of the TMLR 2023 paper "Catastrophic overfitting can be induced with discriminative non-robust features" by Guillermo Ortiz-Jimenez*, Pau de Jorge*, Amartya Sanyal, Adel Bibi, Puneet Dokania, Pascal Frossard, Gregory Rogez, and Philip Torr.

The repository contains code to reproduce the main experiments of the paper and allow for easy experimentation within the same setup. In particular, the repository provides a clean implementation of the main attacks used in the paper, reproducible training loops, and utilities to manipulate the data in the same ways explained in the paper. Furthermore, it contains example scripts to train a PreActResNet18 using different adversarial training methods (e.g., FGSM-AT, PGD-AT, NFGSM...) on different injected and low-passed versions of CIFAR10.

## Dependencies

To run the code, please install all its dependencies by running:

``` sh
$ conda env create -f environment.yml
```

This assumes that you have access to a Linux machine with an NVIDIA GPU.

## Injected features

To reproduce our training runs on different injected versions of CIFAR10, you can run
``` sh
$ python injected_feature_cifar_fgsm_train.py beta=4 epsilon=6
```
where the values of `beta` and `epsilon` can be modified at will to sweep over different training regimes.

## Injected orthogonal features

To reproduce our training runs on different injected versions of CIFAR10 in which the injected features are orthogonal to the data, you can run
``` sh
$ python injected_perp_feature_cifar_fgsm_train.py beta=4 epsilon=6
```
where the values of `beta` and `epsilon` can be modified at will to sweep over different training regimes.

## Low pass experiments

To reproduce our training runs on different low-pass versions of CIFAR10, you can run
``` sh
$ python low_pass_fgsm_train.py bandwidth=16 epsilon=8
```
where the values of `bandwidth` and `epsilon` can be modified at will to sweep over different training regimes.


# Citation
If you found this code useful, please cite our work as
```bibtex
@article{co_features,
  author = {Guillermo Ortiz-Jimenez and Pau de Jorge and Amartya Sanyal and Adel Bibi and Puneet Dokania and Pascal Frossard and Gregory Rogez and Philip Torr},
  title = {Catastrophic overfitting can be induced with discriminative non-robust features},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2023},
}
```

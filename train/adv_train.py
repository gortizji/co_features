import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.utils import Trainer

from .utils import accuracy


class AdvTrainer(Trainer):
    def __init__(
        self, model, epochs, scheduler, optimizer, attacker, eval_attacker=None, save_deltas=False
    ):
        self.model = model
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.attacker = attacker
        self.eval_attacker = eval_attacker
        self.save_deltas = save_deltas
        if self.save_deltas:
            self.deltas = {}
            if self.eval_attacker is not None:
                self.eval_deltas = {}

        super().__init__()

    def train(self, train_loader, test_loader=None):
        step = 0
        for epoch in range(self.epochs):
            self.model.train()

            with tqdm(total=len(train_loader)) as pbar:
                for _, (X, y) in enumerate(train_loader):

                    X, y = self.preprocess_data(X, y)
                    loss, delta, adv_output = self.attacker.attack(model=self.model, X=X, y=y)
                    reg_loss = self.regularized_loss(loss, delta, adv_output, X, y)

                    self.pre_step_train_callback(X, y, reg_loss, delta, adv_output, step)

                    self.optimizer.zero_grad()
                    reg_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.post_step_train_callback(X, y, reg_loss, delta, adv_output, step)

                    log_dict = {
                        "train.adv_loss": reg_loss.item(),
                        "train.adv_acc": accuracy(adv_output, y).item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    pbar.set_description(f"Epoch {epoch}: ")
                    pbar.set_postfix(log_dict)
                    pbar.update()
                    step += 1

            if test_loader is not None:
                evaluate_log_dict = self.evaluate(test_loader)
                self.log(evaluate_log_dict, explicit_print=True)

            self.end_epoch_callback(train_loader, test_loader, epoch)

    def preprocess_data(self, X, y):
        return X.cuda(), y.cuda()

    def regularized_loss(self, loss, delta, adv_output, X, y):
        return loss

    def evaluate(self, test_loader):
        eval_dict = {}
        if isinstance(test_loader, dict):
            for dict_label, loader in test_loader.items():
                result_dict = self.__evaluate(loader, dict_label)
                eval_dict = {**eval_dict, **result_dict}
        else:
            eval_dict = self.__evaluate(test_loader, "test")

        return eval_dict

    def post_step_train_callback(self, X, y, loss, delta, adv_output, step):
        pass

    def pre_step_train_callback(self, X, y, loss, delta, adv_output, step):
        pass

    def end_epoch_callback(self, train_loader, test_loader, epoch):
        pass

    def __evaluate(self, test_loader, dict_label="test"):
        self.model.eval()

        test_acc = 0.0
        test_adv_acc = 0.0
        test_adv_loss = 0.0
        eval_adv_loss = 0.0
        eval_adv_acc = 0.0
        test_loss = 0.0

        if self.save_deltas:
            deltas = []
            if self.eval_attacker is not None:
                eval_deltas = []

        for batch_it, (X, y) in enumerate(test_loader):
            X, y = self.preprocess_data(X, y)

            if self.eval_attacker is None:
                adv_loss, adv_delta, adv_output = self.attacker.attack(self.model, X, y)
                eval_adv_loss = eval_adv_acc = None

                if self.save_deltas:
                    deltas.append(adv_delta.cpu())
            else:
                adv_loss, adv_delta, adv_output = self.attacker.attack(self.model, X, y)
                eval_adv_loss, eval_adv_delta, eval_adv_output = self.eval_attacker.attack(
                    self.model, X, y
                )

                eval_adv_loss += eval_adv_loss.item()
                eval_adv_acc += accuracy(eval_adv_output, y).item()

                if self.save_deltas:
                    deltas.append(adv_delta.cpu())
                    eval_deltas.append(eval_adv_delta.cpu())

            test_adv_loss += adv_loss.item()
            test_adv_acc += accuracy(adv_output, y).item()

            with torch.no_grad():
                output = self.model(X)
                test_loss += F.cross_entropy(output, y).item()
                test_acc += accuracy(output, y).item()

        if self.save_deltas:
            if dict_label in self.deltas.keys():
                self.deltas[dict_label] = torch.concat(
                    [self.deltas[dict_label], torch.concat(deltas)[None, ...]]
                )
            else:
                self.deltas[dict_label] = torch.concat(deltas)[None, ...]

            if self.eval_attacker is not None:
                if dict_label in self.eval_deltas.keys():
                    self.eval_deltas[dict_label] = torch.concat(
                        [self.eval_deltas[dict_label], torch.concat(eval_deltas)[None, ...]]
                    )
                else:
                    self.eval_deltas[dict_label] = torch.concat(eval_deltas)[None, ...]

        return {
            f"{dict_label}.adv_acc": test_adv_acc / (batch_it + 1),
            f"{dict_label}.adv_loss": test_adv_loss / (batch_it + 1),
            f"{dict_label}.eval_adv_acc": eval_adv_acc / (batch_it + 1)
            if eval_adv_acc is not None
            else None,
            f"{dict_label}.eval_adv_loss": eval_adv_loss / (batch_it + 1)
            if eval_adv_loss is not None
            else None,
            f"{dict_label}.acc": test_acc / (batch_it + 1),
            f"{dict_label}.loss": test_loss / (batch_it + 1),
        }


class WarmupAdvTrainer(AdvTrainer):
    def __init__(
        self,
        model,
        epochs,
        scheduler,
        optimizer,
        attacker,
        eval_attacker=None,
        warmup_iterations=None,
    ):
        self.warmup_iterations = warmup_iterations
        super().__init__(model, epochs, scheduler, optimizer, attacker, eval_attacker)

    def train(self, train_loader, test_loader=None):

        iter_count = 0
        for epoch in range(self.epochs):
            self.model.train()

            with tqdm(total=len(train_loader)) as pbar:
                for _, (X, y) in enumerate(train_loader):
                    pbar.set_description(f"Epoch {epoch}: ")
                    X, y = self.preprocess_data(X, y)

                    if self.warmup_iterations is not None:
                        warmup_scale = min(iter_count / self.warmup_iterations, 1.0)
                    else:
                        warmup_scale = 1.0

                    loss, _, adv_output = self.attacker.attack(
                        model=self.model, X=X, y=y, warmup_scale=warmup_scale
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    log_dict = {
                        "train.adv_loss": loss.item(),
                        "train.adv_acc": accuracy(adv_output, y).item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }

                    self.log(log_dict)
                    pbar.update()

                    iter_count += 1

            if test_loader is not None:
                evaluate_log_dict = self.evaluate(test_loader)
                self.log(evaluate_log_dict, explicit_print=True)

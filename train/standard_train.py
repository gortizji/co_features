import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.utils import Trainer, accuracy


class StdTrainer(Trainer):
    def __init__(self, model, epochs, scheduler, optimizer):
        self.model = model
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        super().__init__()

    def train(self, train_loader, test_loader=None):
        step = 0
        for epoch in range(self.epochs):

            self.model.train()
            with tqdm(total=len(train_loader)) as pbar:
                for _, (X, y) in enumerate(train_loader):
                    X, y = X.cuda(), y.cuda()
                    output = self.model(X)
                    loss = F.cross_entropy(output, y)

                    self.pre_step_train_callback(X, y, loss, step)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.post_step_train_callback(X, y, loss, step)

                    log_dict = {
                        "train.loss": loss.item(),
                        "train.acc": accuracy(output, y).item(),
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

    def evaluate(self, test_loader):
        eval_dict = {}
        if isinstance(test_loader, dict):
            for dict_label, loader in test_loader.items():
                result_dict = self.__evaluate(loader, dict_label)
                eval_dict = {**eval_dict, **result_dict}
        else:
            eval_dict = self.__evaluate(test_loader, "test")

        return eval_dict

    def post_step_train_callback(self, X, y, loss, step):
        pass

    def pre_step_train_callback(self, X, y, loss, step):
        pass

    def end_epoch_callback(self, train_loader, test_loader, epoch):
        pass

    def __evaluate(self, test_loader, dict_label="test"):
        self.model.eval()

        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            for batch_it, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                output = self.model(X)
                loss = F.cross_entropy(output, y)
                test_loss += loss.item()
                test_acc += accuracy(output, y).item()

        return {
            f"{dict_label}.acc": test_acc / (batch_it + 1),
            f"{dict_label}.loss": test_loss / (batch_it + 1),
        }

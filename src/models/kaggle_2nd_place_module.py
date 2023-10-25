from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric, LogCoshError

# from torchmetrics.classification.accuracy import Accuracy
import pandas as pd


class Kaggle2ndPlaceLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = model

        # loss function
        self.loss_fn = loss_fn  # torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        # List for saving predictions and targets for test step
        # self.test_step_preds = []
        # self.test_step_targets = []
        # self.test_step_loss = []
        # self.event_no = []

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        pass

    def model_step(self, batch: Any):
        x, y, event_no = batch
        # print(x.shape, pad_mask.shape)
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, y, event_no

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        # self.test_step_preds = []
        # self.test_step_targets = []
        # self.test_step_loss = []
        # self.event_no = []
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)

        # self.test_step_preds.append(preds)
        # self.test_step_targets.append(targets)
        # self.event_no.append(event_no)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    def on_validation_epoch_end(self):
        acc = self.val_loss.compute()  # get current val acc
        self.val_loss_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # all_preds = torch.cat(self.test_step_preds, dim=0).cpu().numpy()
        # all_targets = torch.cat(self.test_step_targets, dim=0).cpu().numpy()
        # all_event_no = torch.cat(self.event_no, dim=0).cpu().numpy()
        # print(self.test_step_targets)
        # print(self.test_step_loss)
        # all_loss = torch.cat(self.test_step_loss,).cpu().numpy()

        # df = pd.DataFrame(
        #     {
        #         "alpha": all_preds[
        #             :, 0
        #         ].flatten(),  # flatten in case preds_np is not 1-D
        #         "beta": all_preds[:, 1].flatten(),
        #         "targets": all_targets.flatten(),  # flatten in case targets_np is not 1-D
        #         "event_no": all_event_no.flatten(),
        #         # 'loss': all_loss.flatten()
        #     }
        # )
        # df = pd.DataFrame({
        #     'predictions': all_preds.flatten(),  # flatten in case preds_np is not 1-D
        #     'targets': all_targets.flatten(),  # flatten in case targets_np is not 1-D
        # })
        # df.to_csv(
        #     "/groups/icecube/moust/storage/test_predictions/inelasticity_predictions_val.csv",
        #     index=False,
        # )

        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def on_test_start(self):
        # self.test_step_preds = []
        # self.test_step_targets = []
        # self.test_step_loss = []
        # self.event_no = []
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)
        print(event_no)
        print(batch_idx)
        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # save predictions and targets for later use in `test_epoch_end()`
        # self.test_step_preds.append(preds)
        # self.test_step_targets.append(targets)
        # self.test_step_loss.append(loss)
        # self.event_no.append(event_no)

        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    # def on_test_epoch_end(self):
    #     # all_preds = torch.cat(self.test_step_preds, dim=0).cpu().numpy()
    #     # all_targets = torch.cat(self.test_step_targets, dim=0).cpu().numpy()
    #     # all_event_no = torch.cat(self.event_no, dim=0).cpu().numpy()

    #     # df = pd.DataFrame(
    #     #     {
    #     #         "alpha": all_preds[:, 0].flatten(),  # flatten in case preds_np is not 1-D
    #     #         "beta": all_preds[:, 1].flatten(),
    #     #         "targets": all_targets.flatten(),  # flatten in case targets_np is not 1-D
    #     #         "event_no": all_event_no.flatten(),
    #     #         # 'loss': all_loss.flatten()
    #     #     }
    #     # )
    #     # df.to_csv(
    #     #     "/groups/icecube/moust/storage/test_predictions/inelasticity_predictions_test.csv",
    #     #     index=False,
    #     # )
    #     pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Kaggle2ndPlaceLitModule(None, None, None)

from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric, LogCoshError

# from torchmetrics.classification.accuracy import Accuracy
import pandas as pd


class Kaggle2ndPlaceLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = model

        # loss function
        self.loss_fn = loss_fn

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        pass

    def model_step(self, batch: Any):
        x, y, event_no = batch
        # print(x.shape, pad_mask.shape)
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, y, event_no

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    def on_validation_epoch_end(self):
        acc = self.val_loss.compute()  # get current val acc
        self.val_loss_best(acc)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def on_test_start(self):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, event_no = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets, "event_no": event_no}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Kaggle2ndPlaceLitModule(None, None, None)

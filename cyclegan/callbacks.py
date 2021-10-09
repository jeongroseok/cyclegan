import torch
import torchvision
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard.writer import SummaryWriter
from wandb.sdk.wandb_run import Run

from .models.cyclegan import CycleGAN


class TranslationVisualization_TensorBoard(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: CycleGAN) -> None:
        writer: SummaryWriter = trainer.logger.experiment
        dataloader = trainer.train_dataloader

        for x, y in dataloader:
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            y_hat = pl_module.generator_g(x)
            x_hat = pl_module.generator_f(y)
            images = torch.cat((x, y, y_hat, x_hat))
            break
        grid = torchvision.utils.make_grid(images, 2, normalize=True)

        str_title = f"{pl_module.__class__.__name__}"

        writer.add_image(str_title, grid, global_step=trainer.global_step)


class TranslationVisualization_WanDB(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: CycleGAN) -> None:
        run: Run = trainer.logger.experiment
        dataloader = trainer.train_dataloader

        for x, y in dataloader:
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            y_hat = pl_module.generator_g(x)
            x_hat = pl_module.generator_f(y)
            images = [
                wandb.Image(x, caption="Real X"),
                wandb.Image(y, caption="Real Y"),
                wandb.Image(y_hat, caption="Fake Y"),
                wandb.Image(x_hat, caption="Fake X"),
            ]
            break
        run.log({"examples": images}, step=trainer.global_step)

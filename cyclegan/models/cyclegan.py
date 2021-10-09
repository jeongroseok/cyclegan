from typing import Tuple
import pytorch_lightning as pl
import torch

from .components import Generator, Discriminator


class CycleGAN(pl.LightningModule):
    class __HPARAMS:
        lr: float
        lambda_id: float
        lambda_cycle: float

    hparams: __HPARAMS

    def __init__(
        self,
        img_dim: Tuple[int, int, int] = (1, 28, 28),
        residual_blocks: int = 4,
        lr=2e-4,
        lambda_id=0.5,
        lambda_cycle=1,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion_adv = torch.nn.BCELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_id = torch.nn.L1Loss()

        self.generator_g = Generator(img_dim, residual_blocks)
        self.generator_f = Generator(img_dim, residual_blocks)
        self.discriminator_x = Discriminator(img_dim)
        self.discriminator_y = Discriminator(img_dim)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_gen = torch.optim.Adam(
            list(self.generator_g.parameters()) + list(self.generator_f.parameters()),
            lr=lr,
        )
        opt_disc_x = torch.optim.Adam(self.discriminator_x.parameters(), lr=lr,)
        opt_disc_y = torch.optim.Adam(self.discriminator_x.parameters(), lr=lr,)
        return [opt_gen, opt_disc_x, opt_disc_y]

    def _generate_samples(self, x, y):
        y_hat = self.generator_g(x)
        x_hat = self.generator_f(y)
        return x_hat, y_hat

    def _calc_identity_loss(self, x, x_hat, y, y_hat):
        loss = self.criterion_id(x_hat, x) + self.criterion_id(y_hat, y)
        return loss / 2

    def _calc_generator_loss(self, x_hat, y_hat):
        d_x = self.discriminator_x(x_hat)
        d_y = self.discriminator_y(y_hat)
        loss_x = self.criterion_adv(d_x, torch.ones_like(d_x))
        loss_y = self.criterion_adv(d_y, torch.ones_like(d_y))
        return (loss_x + loss_y) / 2

    def _calc_cycle_loss(self, x, x_hat, y, y_hat):
        loss_x = self.criterion_cycle(self.generator_f.forward(y_hat), x)
        loss_y = self.criterion_cycle(self.generator_g.forward(x_hat), y)
        return (loss_x + loss_y) / 2

    def _generation_step(self, x, y):
        x_hat, y_hat = self._generate_samples(y, x)  # g(y) ~ y, f(x) ~ x
        loss_id = self._calc_identity_loss(x, x_hat, y, y_hat)

        x_hat, y_hat = self._generate_samples(x, y)  # Dy(g(x)) = True, Dx(f(y)) = True
        loss_adv = self._calc_generator_loss(x_hat, y_hat)
        loss_cycle = self._calc_cycle_loss(x, x_hat, y, y_hat)

        loss = (
            loss_adv
            + (loss_cycle * self.hparams.lambda_cycle)
            + (loss_id * self.hparams.lambda_id)
        )

        return loss, loss_id, loss_adv, loss_cycle

    def _discrimination_x_step(self, x, y):
        x_hat = self.generator_f(y)
        d_real = self.discriminator_x(x)
        d_fake = self.discriminator_x(x_hat)
        
        loss_real = self.criterion_adv(d_real, torch.ones_like(d_real))
        loss_fake = self.criterion_adv(d_fake, torch.zeros_like(d_fake))
        loss = (loss_real + loss_fake) / 2

        return loss, loss_real, loss_fake

    def _discrimination_y_step(self, x, y):
        y_hat = self.generator_g(x)
        d_real = self.discriminator_y(y)
        d_fake = self.discriminator_y(y_hat)
        
        loss_real = self.criterion_adv(d_real, torch.ones_like(d_real))
        loss_fake = self.criterion_adv(d_fake, torch.zeros_like(d_fake))
        loss = (loss_real + loss_fake) / 2

        return loss, loss_real, loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        if optimizer_idx == 0:
            loss, loss_id, loss_adv, loss_cycle = self._generation_step(x, y)
            self.log(f"train/gen/id", loss_id)
            self.log(f"train/gen/adv", loss_adv)
            self.log(f"train/gen/cycle", loss_cycle)
            return loss
        elif optimizer_idx == 1:
            loss, loss_real, loss_fake = self._discrimination_x_step(x, y)
            self.log(f"train/disc_x/adv_real", loss_real)
            self.log(f"train/disc_x/adv_fake", loss_fake)
            return loss
        elif optimizer_idx == 2:
            loss, loss_real, loss_fake = self._discrimination_y_step(x, y)
            self.log(f"train/disc_y/adv_real", loss_real)
            self.log(f"train/disc_y/adv_fake", loss_fake)
            return loss

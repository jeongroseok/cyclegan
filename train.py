import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pl_examples import _DATASETS_PATH

from cyclegan.datamodules.apple2orange import Apple2OrangeDataModule
from cyclegan.models.cyclegan import CycleGAN
from cyclegan.callbacks import TranslationVisualization_WanDB


def main(args=None):
    datamodule = Apple2OrangeDataModule(
        _DATASETS_PATH,
        num_workers=1,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        max_samples=25,
    )

    model = CycleGAN(img_dim=datamodule.dims, residual_blocks=4,)

    wandb_logger = WandbLogger(project="cyclegan")

    callbacks = [
        TranslationVisualization_WanDB(),
    ]

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=100,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

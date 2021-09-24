from cyclegan.datamodules.apple2orange import Apple2OrangeDataModule
from pl_examples import _DATASETS_PATH

def main():
    dm = Apple2OrangeDataModule(_DATASETS_PATH, num_workers=0)
    dm.setup()

    dl_train = dm.train_dataloader()
    for batch in dl_train:
        batch

if __name__ == "__main__":
    main()

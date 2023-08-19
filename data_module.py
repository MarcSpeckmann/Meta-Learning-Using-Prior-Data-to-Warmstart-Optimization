import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from filelock import FileLock
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DEEP_WOODS_LINK = "http://ml.informatik.uni-freiburg.de/~biedenka/dwx_compressed.tar.gz"
DEEP_WOODS_DEFAULT_DIR = "deepweedsx"
DEEP_WOODS_DEFAULT_TARBALL_PATH = "processed_data.tar"


class DeepWeedsDataModule(pl.LightningDataModule):
    """
    This class is a PyTorch Lightning DataModule for the DeepWeedsX dataset.
    It downloads the dataset, splits it into train, validation and test sets and
    returns the corresponding PyTorch DataLoaders.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        # Saves all arguments passed to the constructor as hyperparameters (self.hparams)
        self.save_hyperparameters()
        # Define the pre-processing steps for the images
        self.pre_processing = transforms.Compose(
            [
                transforms.Resize(self.hparams.img_size),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # To allow the use of multiple workers for data loading, we need to use a temporary directory.
        # Especially when using multiple GPUs, the data loading can become a bottleneck.
        if self.hparams.load_data_on_every_trial:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.data_path = Path(self.temp_dir.name)
        else:
            self.data_path = self.hparams.data_path

    def prepare_data(self):
        """Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single
        process, so you can safely add your downloading logic within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device
        """
        datadir = Path(os.path.join(self.data_path, DEEP_WOODS_DEFAULT_DIR))
        tar_path = Path(os.path.join(datadir, DEEP_WOODS_DEFAULT_TARBALL_PATH))

        with FileLock("./.data.lock"):
            if not datadir.exists():
                datadir.mkdir(parents=True)

            self._download_deepweeds(url=DEEP_WOODS_LINK, dest=tar_path)

            self._unpack_tarball(tarball=tar_path, dest=datadir)

    def setup(self, stage: str) -> None:
        """
        Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        # Load the dataset and split into train and test sets
        train_dataset, test_dataset = self.load_deep_woods(
            datadir=Path(os.path.join(self.data_path, DEEP_WOODS_DEFAULT_DIR)),
            balanced=self.hparams.balanced,
        )

        # Split the train_dataset into train and validation sets
        val_size = int(self.hparams.train_val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.test_dataset = test_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """An iterable or collection of iterables specifying training samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~lightning.pytorch.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying validation samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~lightning.pytorch.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`
        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying test samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """An iterable or collection of iterables specifying prediction samples.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.
        """
        pass

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        if self.hparams.load_data_on_every_trial:
            self.temp_dir.cleanup()
        return super().teardown(stage)

    def _unpack_tarball(
        self,
        tarball: Path = DEEP_WOODS_DEFAULT_TARBALL_PATH,
        dest: Path = DEEP_WOODS_DEFAULT_DIR,
    ) -> None:
        """Unpacks the tarball containing the DeepWeeds.

        Code from the project template.

        Args:
            tarball (Path, optional): _description_. Defaults to DEEP_WOODS_DEFAULT_TARBALL_PATH.
            dest (Path, optional): _description_. Defaults to DEEP_WOODS_DEFAULT_DIR.
        """
        dir_contents = list(dest.iterdir())
        if len(dir_contents) > 1:
            logging.debug(
                "Already unpacked most likely with %s items in %s. \n%s",
                str(len(dir_contents)),
                dest,
                dir_contents,
            )
        logging.debug("Unpacking %s", tarball)
        tar = tarfile.open(name=tarball, mode="r:gz")
        tar.extractall(path=dest)

    def _download_deepweeds(
        self, url: str = DEEP_WOODS_LINK, dest: Path = DEEP_WOODS_DEFAULT_TARBALL_PATH
    ) -> Path:
        """Downloads the DeepWeeds dataset.

        Code from the project template.

        Args:
            url (str, optional): _description_. Defaults to DEEP_WOODS_LINK.
            dest (Path, optional): _description_. Defaults to DEEP_WOODS_DEFAULT_TARBALL_PATH.

        Returns:
            Path: _description_
        """
        if dest.exists():
            logging.debug("Already found file at %s", dest)
            return dest

        logging.debug("Downloading from %s to %s", url, dest)
        dest.parent.mkdir(exist_ok=True, parents=True)

        with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
            shutil.copyfileobj(response, f)

        logging.debug("Download finished at %s", dest)
        return dest

    def load_deep_woods(
        self,
        datadir: Path,
        balanced: bool = False,
    ) -> Tuple[ImageFolder, ImageFolder]:
        """Load the DeepWeeds dataset.

        Code from the project template

        :param balanced:
            Whether to load the balanced dataset or not.
        :param resize:
            What to resize the image to
        :param transform:
            The transformation to apply to images.

        :return: The train and test datasets.
        """
        suffix = "" if not balanced else "_balanced"

        train_path = datadir / f"train{suffix}"
        test_path = datadir / f"test{suffix}"

        train_dataset = ImageFolder(root=str(train_path), transform=self.pre_processing)
        test_dataset = ImageFolder(root=str(test_path), transform=self.pre_processing)

        return train_dataset, test_dataset


if __name__ == "__main__":
    data_model = DeepWeedsDataModule(
        batch_size=32,
        img_size=(32, 32),
        balanced=True,
        train_val_split=0.1,
        num_workers=4,
        data_path="./data",
        load_data_on_every_trial=False,
        seed=42,
    )

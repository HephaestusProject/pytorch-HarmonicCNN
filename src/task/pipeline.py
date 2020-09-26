import pickle

from omegaconf import DictConfig
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from ..data import MTATDataset

class DataPipeline(LightningDataModule):
    def __init__(self, pipline_config: DictConfig) -> None:
        super(DataPipeline, self).__init__()
        self.pipeline_config = pipline_config
        self.dataset_builder = MTATDataset

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                                                        self.dataset_builder,
                                                        self.pipeline_config.dataset.path,
                                                        "TRAIN",
                                                        self.pipeline_config.dataset.input_length
                                                        )

            self.val_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                        self.pipeline_config.dataset.path,
                                                        "VALID",
                                                        self.pipeline_config.dataset.input_length)

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                        self.pipeline_config.dataset.path,
                                                        "TEST",
                                                        self.pipeline_config.dataset.input_length)

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.train_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=True,
                                           shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.val_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=True,
                                           shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.test_dataset,
                                           batch_size=self.pipeline_config.dataloader.params.batch_size,
                                           num_workers=self.pipeline_config.dataloader.params.num_workers,
                                           drop_last=True,
                                           shuffle=False)

    @classmethod
    def get_dataset(cls, dataset_builder:Callable, root, split, length) -> Dataset:
        dataset = dataset_builder(root, split, length)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool,
                       **kwargs) -> DataLoader:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          **kwargs)
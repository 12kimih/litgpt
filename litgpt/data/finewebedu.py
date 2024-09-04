# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

import tiktoken
import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class FineWebEdu(DataModule):
    """The FineWebEdu data module for pretraining."""

    data_path: Union[str, Path] = Path("data/fineweb-edu")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        super().__init__()
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048) -> None:
        self.tokenizer = tiktoken.get_encoding("p50k_base")
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(f"The FineWebEdu data path points to an S3 location: {self.data_path}. Skipping preprocessing.")
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found FineWebEdu train and val dir: {self.data_path}. Skipping preprocessing.")
            return

        dataset = load_dataset(path="HuggingFaceFW/fineweb-edu", name="sample-350BT", num_proc=os.cpu_count() // 2)

        # Split the data in training and validation
        split_dataset = dataset["train"].train_test_split(test_size=self.val_split_fraction, seed=self.seed, shuffle=True)
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

        def tokenize(data: Dataset, index: int):
            ids = self.tokenizer.encode_ordinary(data[index]["text"])
            ids.append(self.tokenizer.eot_token)
            ids_pt = torch.tensor(ids, dtype=torch.int32)
            yield ids_pt

        optimize(
            fn=partial(tokenize, split_dataset["train"]),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=os.cpu_count() // 2,
            chunk_bytes="200MB",
        )
        optimize(
            fn=partial(tokenize, split_dataset["val"]),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=os.cpu_count() // 2,
            chunk_bytes="200MB",
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        train_dataloader = StreamingDataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True)
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        val_dataloader = StreamingDataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True)
        return val_dataloader

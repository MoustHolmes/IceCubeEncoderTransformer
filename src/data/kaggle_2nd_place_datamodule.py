import sqlite3
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from typing import Any, Callable, List, Dict, Optional, Sequence, Tuple, Union, Iterator
from lightning import LightningDataModule
import numpy as np
import random

def combine_dom_types_and_rde(dom_type, rde):
    # pDom dom with low efficiency
    pdom_low_qe = ((dom_type == 20) & (rde == 1)).long() * 0
    # pDOM dom with high efficiency
    pdom_high_qe = ((dom_type == 20) & (rde == 1.35)).long() * 1
    # pDOM upgrade == 110
    pdom_upgrade = (dom_type == 110).long() * 2
    # D-EGG == 120
    d_egg = (dom_type == 120).long() * 3
    # mDOM == 130
    mdom = (dom_type == 130).long() * 4

    return pdom_low_qe + pdom_high_qe + pdom_upgrade + d_egg + mdom

class ChunkDataset(Dataset):
    """
    PyTorch dataset for loading chunked data from an SQLite database.
    This dataset retrieves pulsemap and truth data for each event from the database.

    Args:
        db_filename (str): Filename of the SQLite database.
        csv_filenames (list of str): List of CSV filenames containing event numbers.
        pulsemap_table (str): Name of the table containing pulsemap data.
        truth_table (str): Name of the table containing truth data.
        truth_variable (str): Name of the variable to query from the truth table.
        feature_variables (list of str): List of variable names to query from the pulsemap table.
    """

    def __init__(
        self,
        db_path: str,
        chunk_csvs: List[str],
        pulsemap: str,
        truth_table: str,
        target_cols: str,
        input_cols: List[str]
    ) -> None:
        self.conn = sqlite3.connect(db_path)  # Connect to the SQLite database
        self.c = self.conn.cursor()
        event_nos = np.array([])
        for csv_filename in chunk_csvs:
            # df = pd.read_csv(csv_filename)
            np.append(event_nos,(pd.read_csv(csv_filename)['event_no'].to_numpy()))  # Collect event numbers from CSV files
        # for csv_filename in chunk_csvs:
        #     df = pd.read_csv(csv_filename)
        #     self.event_nos.extend(df['event_no'].tolist())
        self.length = len(event_nos)
        self.pulsemap = pulsemap  # Name of the table containing pulsemap data
        self.truth_table = truth_table  # Name of the table containing truth data
        self.target_cols = target_cols  # Name of the variable to query from the truth table
        self.input_cols = input_cols  # List of variable names to query from the pulsemap table


    def __len__(self) -> int:
        return self.length#len(self.event_nos)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        event_no = idx # self.event_nos[idx]

        # Query the truth variable for the given event number
        self.c.execute(f"SELECT {self.target_cols} FROM {self.truth_table} WHERE event_no = ?", (event_no,))
        truth_value = self.c.fetchone()[0]
        if self.target_cols == "energy":
            truth_value = np.log10(truth_value)
        

        pos_cols = ['dom_x', 'dom_y', 'dom_z']

        rde_index = self.input_cols.index('rde')
        dom_type_index = self.input_cols.index('dom_type')
        pos_indices = [self.input_cols.index(col) for col in pos_cols]
        rest_indices = [i for i in range(len(self.input_cols)) if i not in [rde_index, dom_type_index] + pos_indices]

        input_query = ', '.join(self.input_cols)
        # Query the feature variables from the pulsemap table for the given event number
        self.c.execute(f"SELECT {input_query} FROM {self.pulsemap} WHERE event_no = ?", (event_no,))
        pulsemap_data_rows = self.c.fetchall()
    
        # Convert pulsemap_data_rows into a dictionary of tensors
        
        pulsemap_data = {self.input_cols[i]: torch.tensor( [row[i] for row in pulsemap_data_rows], dtype=torch.float32)
                        for i in rest_indices}
        
        # Get the necessary data for combined_dom_type and pos
        dom_type_data = [row[dom_type_index] for row in pulsemap_data_rows] #[row[i] for row in pulsemap_data_rows for i in combined_dom_type_indices]
        rde_data = [row[rde_index] for row in pulsemap_data_rows]
        pos_data = [[row[i] for row in pulsemap_data_rows] for i in pos_indices]

        pulsemap_data["combined_dom_type"] = combine_dom_types_and_rde(torch.tensor(dom_type_data, dtype=torch.float32),
                                                                    torch.tensor(rde_data, dtype=torch.float32))
        pulsemap_data["pos"] = torch.stack([torch.tensor(col_data, dtype=torch.float32) for col_data in pos_data], dim=1)
        pulsemap_data["L0"] = torch.tensor(len(pulsemap_data_rows), dtype=torch.int32)
        pulsemap_data["charge"] = torch.log10(pulsemap_data["charge"])/3.0
        pulsemap_data["dom_time"] = (pulsemap_data["dom_time"]-1e4)/3e4

        return pulsemap_data, torch.tensor(truth_value, dtype=torch.float32), torch.tensor(event_no, dtype=torch.int32)
    


class ChunkSampler(Sampler):
    """
    PyTorch sampler for creating chunks from event numbers.

    Args:
        csv_filenames (List[str]): List of CSV filenames containing event numbers.
        batch_sizes (List[int]): List of batch sizes for each CSV file.
    """

    def __init__(
        self, 
        chunk_csvs: List[str], 
        batch_sizes: List[int]
    ) -> None:


        # self.event_nos = np.array([])
        # for csv_filename, batch_size in zip(chunk_csvs, batch_sizes):
        #     event_nos = pd.read_csv(csv_filename)['event_no'].values
        #     split_event_nos = np.array([event_nos[i:i + batch_size] for i in range(0, len(event_nos), batch_size)])
        #     self.event_nos = np.concatenate((self.event_nos, split_event_nos))

        self.event_nos = []
        for csv_filename, batch_size in zip(chunk_csvs, batch_sizes):
            event_nos = pd.read_csv(csv_filename)['event_no'].tolist()
            self.event_nos.extend([event_nos[i:i + batch_size] for i in range(0, len(event_nos), batch_size)])
    def __iter__(self) -> Iterator:
        random.shuffle(self.event_nos)
        return iter(self.event_nos)

    def __len__(self) -> int:
        return len(self.event_nos)
    

def collate_fn(batch):
    """
    This collate function is specifically designed for the dataset that
    returns a dictionary of tensors. It will pad the sequences to the same
    length and concatenate along the batch dimension given a list of such 
    dictionaries (i.e., a batch).
    """
    batch_keys = batch[0][0].keys()
    collated_batch = {}

    max_len = max(max(item[0][key].size(0) for item in batch) for key in batch_keys if key != 'L0')

    for key in batch_keys:
        if key != 'L0':
            # Pad the sequences to the same length and stack along a new batch dimension
            collated_batch[key] = torch.stack([torch.cat([item[0][key], item[0][key].new_zeros(max_len - item[0][key].size(0), item[0][key].size(1)) if len(item[0][key].shape) > 1 else item[0][key].new_zeros(max_len - item[0][key].size(0))]) for item in batch])
        else:
            # If the key is 'L0', simply collect the values into a list
            collated_batch[key] = torch.tensor([item[0][key] for item in batch])

    # Create a mask that indicates where the original sequence ends and the padding begins
    collated_batch['mask'] = collated_batch['L0'].new_ones((len(batch), max_len)).bool()
    for i, l0 in enumerate(collated_batch['L0']):
        collated_batch['mask'][i, l0:] = False
    # Stack all target tensors along a new batch dimension
    targets = torch.stack([item[1] for item in batch])
    event_nos = torch.tensor([item[2] for item in batch])
    return collated_batch, targets, event_nos

class Kaggle2ndPlaceDatamodule(LightningDataModule):
    def __init__(
        self,
        db_path: str,
        pulsemap: str,
        input_cols: List[str],
        target_cols: str,
        chunk_csv_train: List[str],
        chunk_csv_test: List[str],
        chunk_csv_val: List[str],
        batch_sizes: List[int],
        truth_table: str = "truth",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            # csv_filenames_train = [os.path.join(self.hparams.csv_folder, "train", f"output_{i}.csv") for i in range(1, 8)]
            # csv_filenames_val = [os.path.join(self.hparams.csv_folder, "val", f"output_{i}.csv") for i in range(1, 8)]
            # csv_filenames_test = [os.path.join(self.hparams.csv_folder, "test", f"output_{i}.csv") for i in range(1, 8)]
            self.data_train = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_train,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )
            self.data_val = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_val,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )
            self.data_test = ChunkDataset(
                db_path=self.hparams.db_path,
                chunk_csvs=self.hparams.chunk_csv_test,
                pulsemap=self.hparams.pulsemap,
                truth_table=self.hparams.truth_table,
                target_cols=self.hparams.target_cols,
                input_cols=self.hparams.input_cols
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs=self.hparams.chunk_csv_train, batch_sizes=self.hparams.batch_sizes) 
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs=self.hparams.chunk_csv_val, batch_sizes=self.hparams.batch_sizes) 
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
            batch_sampler=ChunkSampler(chunk_csvs = self.hparams.chunk_csv_test, batch_sizes=self.hparams.batch_sizes)
        )

    def teardown(self, stage: Optional[str] = None):
        # self.data_test.close_connection()
        # self.data_val.close_connection()
        # self.data_train.close_connection()
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass




if __name__ == "__main__":
    _ = Kaggle2ndPlaceDatamodule()
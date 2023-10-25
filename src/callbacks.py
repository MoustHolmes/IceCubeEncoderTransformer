import pandas as pd
import torch
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import sqlite3

from lightning.pytorch.callbacks import Callback


class BetaPredictionWriterCallback(Callback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_test_start(self, trainer, pl_module):
        self.predictions = []
        self.targets = []
        self.event_nos = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Collect predictions, targets and event numbers from the batch and store them
        self.predictions.append(outputs["preds"])
        self.targets.append(outputs["targets"])
        self.event_nos.append(outputs["event_no"])

    def on_test_end(self, trainer, pl_module):
        # Concatenate all batch predictions and save them as a CSV file
        all_preds = torch.cat(self.predictions, dim=0).cpu().numpy()
        all_targets = torch.cat(self.targets, dim=0).cpu().numpy()
        all_event_nos = torch.cat(self.event_nos, dim=0).cpu().numpy()
        df = pd.DataFrame(
            {
                "alpha": all_preds[:, 0].flatten(),
                "beta": all_preds[:, 1].flatten(),
                "targets": all_targets.flatten(),
                "event_no": all_event_nos.flatten(),
            }
        )

        df.to_csv(self.save_path, index=False)


# class AttentionWeightsWriterCallback(Callback):
#     def __init__(self, save_path):
#         self.save_path = save_path

#     def remove_padding(self, matrix, mask, attn_type):
#         # Find the indices of the non-padded elements
#         # Remove the first element if the attention type is relative due to lack or cls token
#         if attn_type == "rel_attn":
#             non_padded_indices = np.where(mask[1:] != float("-inf"))[0]
#         else:
#             non_padded_indices = np.where(mask != float("-inf"))[0]

#         return matrix[np.ix_(non_padded_indices, non_padded_indices)]

#     def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
#         batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
#         for event_idx in range(batch_size):
#             for layer_no in range(n_layers):
#                 for head_no in range(n_heads):
#                     matrix = (
#                         attn_batch[event_idx, layer_no, head_no]
#                         .cpu()
#                         .numpy()
#                         .astype(np.float64)
#                     )  # Convert PyTorch tensor to NumPy Float64 array
#                     matrix = self.remove_padding(
#                         matrix, mask[event_idx].cpu().numpy(), attn_type
#                     )
#                     self.data.append(
#                         {
#                             "event_number": batch_event_no[event_idx]
#                             .cpu()
#                             .numpy()
#                             .item(),
#                             "attn_type": attn_type,
#                             "layer_no": layer_no,
#                             "head_no": head_no,
#                             "matrix": matrix.tobytes(),
#                         }
#                     )

#     def on_test_start(self, trainer, pl_module):
#         self.data = []

#     def on_test_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         x, y, event_no = batch
#         weights = pl_module.model.get_attn_weights(x)

#         self.process_batch(
#             weights["rel_attn_weights"], event_no, weights["mask"], "rel_attn"
#         )
#         self.process_batch(weights["attn_weights"], event_no, weights["mask"], "attn")

#     def on_test_end(self, trainer, pl_module):
#         df = pd.DataFrame(self.data)
#         df.to_parquet(self.save_path)


# class AttentionWeightsWriterCallback(Callback):
#     def __init__(self, save_path):
#         self.save_path = save_path

#     def remove_padding(self, matrix, mask, attn_type):
#         if attn_type == "rel_attn":
#             non_padded_indices = np.where(mask[1:] != float("-inf"))[0]
#         else:
#             non_padded_indices = np.where(mask != float("-inf"))[0]

#         return matrix[np.ix_(non_padded_indices, non_padded_indices)]

#     def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
#         batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
#         batch_data = []
#         for event_idx in range(batch_size):
#             for layer_no in range(n_layers):
#                 for head_no in range(n_heads):
#                     matrix = (
#                         attn_batch[event_idx, layer_no, head_no]
#                         .cpu()
#                         .numpy()
#                         .astype(np.float64)
#                     )
#                     matrix = self.remove_padding(
#                         matrix, mask[event_idx].cpu().numpy(), attn_type
#                     )
#                     batch_data.append(
#                         {
#                             "event_number": batch_event_no[event_idx]
#                             .cpu()
#                             .numpy()
#                             .item(),
#                             "attn_type": attn_type,
#                             "layer_no": layer_no,
#                             "head_no": head_no,
#                             "matrix": matrix.tobytes(),
#                         }
#                     )
#         return pd.DataFrame(batch_data)

#     def on_test_start(self, trainer, pl_module):
#         self.temp_dir = tempfile.mkdtemp()
#         self.temp_files = []

#     def on_test_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         x, y, event_no = batch
#         weights = pl_module.model.get_attn_weights(x)

#         rel_attn_df = self.process_batch(
#             weights["rel_attn_weights"], event_no, weights["mask"], "rel_attn"
#         )
#         attn_df = self.process_batch(
#             weights["attn_weights"], event_no, weights["mask"], "attn"
#         )

#         temp_file = os.path.join(self.temp_dir, f"batch_{batch_idx}.parquet")
#         pd.concat([rel_attn_df, attn_df]).to_parquet(temp_file)
#         self.temp_files.append(temp_file)

#     def on_test_end(self, trainer, pl_module):
#         # Concatenate all temporary parquet files and save the result to the final path
#         df = pd.concat(
#             [pd.read_parquet(temp_file) for temp_file in self.temp_files],
#             ignore_index=True,
#         )
#         df.to_parquet(self.save_path)

#         # Clean up temporary files and directory
#         for temp_file in self.temp_files:
#             os.remove(temp_file)
#         os.rmdir(self.temp_dir)


# class AttentionWeightsWriterCallback(Callback):
#     def __init__(self, save_path):
#         self.save_path = save_path

#     def remove_padding(self, matrix, mask, attn_type):
#         if attn_type == "rel_attn":
#             non_padded_indices = np.where(mask[1:] != float("-inf"))[0]
#         else:
#             non_padded_indices = np.where(mask != float("-inf"))[0]

#         return matrix[np.ix_(non_padded_indices, non_padded_indices)]

#     def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
#         batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
#         batch_data = []
#         for event_idx in range(batch_size):
#             for layer_no in range(n_layers):
#                 for head_no in range(n_heads):
#                     matrix = (
#                         attn_batch[event_idx, layer_no, head_no]
#                         .cpu()
#                         .numpy()
#                         .astype(np.float64)
#                     )
#                     matrix = self.remove_padding(
#                         matrix, mask[event_idx].cpu().numpy(), attn_type
#                     )
#                     batch_data.append(
#                         {
#                             "event_number": batch_event_no[event_idx]
#                             .cpu()
#                             .numpy()
#                             .item(),
#                             "attn_type": attn_type,
#                             "layer_no": layer_no,
#                             "head_no": head_no,
#                             "matrix": matrix.tobytes(),
#                         }
#                     )
#         return pd.DataFrame(batch_data)

#     def on_test_start(self, trainer, pl_module):
#         # Initialize a Parquet writer with no data to start appending batches
#         self.writer = None

#     def on_test_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         x, y, event_no = batch
#         weights = pl_module.model.get_attn_weights(x)

#         rel_attn_df = self.process_batch(
#             weights["rel_attn_weights"], event_no, weights["mask"], "rel_attn"
#         )
#         attn_df = self.process_batch(
#             weights["attn_weights"], event_no, weights["mask"], "attn"
#         )

#         # Concatenate the DataFrames for the current batch
#         batch_df = pd.concat([rel_attn_df, attn_df])

#         # Convert the DataFrame to an Arrow Table
#         table = pa.Table.from_pandas(batch_df, preserve_index=False)

#         # If the writer is not initialized, initialize it with the schema of the Arrow Table
#         if self.writer is None:
#             self.writer = pq.ParquetWriter(self.save_path, table.schema)

#         # Write the Arrow Table as a new row group to the existing Parquet file
#         self.writer.write_table(table)

#     def on_test_end(self, trainer, pl_module):
#         # Close the Parquet writer
#         if self.writer:
#             self.writer.close()


# class AttentionWeightsWriterCallback(Callback):
#     def __init__(self, save_path):
#         self.db_path = save_path  # db_path

#     def remove_padding(self, matrix, mask, attn_type):
#         if attn_type == "rel_attn":
#             non_padded_indices = np.where(mask[1:] != float("-inf"))[0]
#         else:
#             non_padded_indices = np.where(mask != float("-inf"))[0]

#         return matrix[np.ix_(non_padded_indices, non_padded_indices)]

#     def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
#         batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
#         batch_data = []
#         for event_idx in range(batch_size):
#             for layer_no in range(n_layers):
#                 for head_no in range(n_heads):
#                     matrix = (
#                         attn_batch[event_idx, layer_no, head_no]
#                         .cpu()
#                         .numpy()
#                         .astype(np.float64)
#                     )
#                     matrix = self.remove_padding(
#                         matrix, mask[event_idx].cpu().numpy(), attn_type
#                     )
#                     batch_data.append(
#                         {
#                             "event_number": batch_event_no[event_idx]
#                             .cpu()
#                             .numpy()
#                             .item(),
#                             "attn_type": attn_type,
#                             "layer_no": layer_no,
#                             "head_no": head_no,
#                             "matrix": matrix.tobytes(),
#                         }
#                     )
#         return pd.DataFrame(batch_data)

#     def on_test_start(self, trainer, pl_module):
#         # Create a new SQLite database and create a table for attention weights
#         self.conn = sqlite3.connect(self.db_path)
#         self.cursor = self.conn.cursor()
#         self.cursor.execute(
#             """CREATE TABLE IF NOT EXISTS attention_weights (
#                                     id INTEGER PRIMARY KEY,
#                                     event_number INTEGER,
#                                     attn_type TEXT,
#                                     layer_no INTEGER,
#                                     head_no INTEGER,
#                                     matrix BLOB)"""
#         )
#         self.conn.commit()

#     def on_test_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         x, y, event_no = batch
#         weights = pl_module.model.get_attn_weights(x)

#         rel_attn_df = self.process_batch(
#             weights["rel_attn_weights"], event_no, weights["mask"], "rel_attn"
#         )
#         attn_df = self.process_batch(
#             weights["attn_weights"], event_no, weights["mask"], "attn"
#         )

#         # Concatenate the DataFrames for the current batch and insert into SQLite database
#         batch_df = pd.concat([rel_attn_df, attn_df])
#         for idx, row in batch_df.iterrows():
#             self.cursor.execute(
#                 """INSERT INTO attention_weights (event_number, attn_type, layer_no, head_no, matrix)
#                                    VALUES (?, ?, ?, ?, ?)""",
#                 (
#                     row["event_number"],
#                     row["attn_type"],
#                     row["layer_no"],
#                     row["head_no"],
#                     row["matrix"],
#                 ),
#             )
#         self.conn.commit()

#     def on_test_end(self, trainer, pl_module):
#         # Close the SQLite database connection
#         self.conn.close()


class AttentionWeightsWriterCallback(Callback):
    def __init__(self, save_path):
        self.db_path = save_path  # db_path

    def remove_padding(self, matrix, mask, attn_type):
        if attn_type == "rel_attn":
            non_padded_indices = np.where(mask[1:] != float("-inf"))[0]
        else:
            non_padded_indices = np.where(mask != float("-inf"))[0]

        return matrix[np.ix_(non_padded_indices, non_padded_indices)]

    def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
        batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
        batch_data = []
        for event_idx in range(batch_size):
            for layer_no in range(n_layers):
                for head_no in range(n_heads):
                    matrix = (
                        attn_batch[event_idx, layer_no, head_no]
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    matrix = self.remove_padding(
                        matrix, mask[event_idx].cpu().numpy(), attn_type
                    )
                    batch_data.append(
                        {
                            "event_number": batch_event_no[event_idx]
                            .cpu()
                            .numpy()
                            .item(),
                            "attn_type": attn_type,
                            "layer_no": layer_no,
                            "head_no": head_no,
                            "matrix": matrix.tobytes(),
                        }
                    )
        return pd.DataFrame(batch_data)

    def on_test_start(self, trainer, pl_module):
        # Create a new SQLite database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # As the number of layers and heads are not known at this point,
        # we will create the table without specifying the matrix columns.
        # The matrix columns will be added dynamically in the on_test_batch_end method.
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS attention_weights (
                                    event_number INTEGER PRIMARY KEY)"""
        )
        self.conn.commit()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y, event_no = batch
        weights = pl_module.model.get_attn_weights(x)

        rel_attn_df = self.process_batch(
            weights["rel_attn_weights"], event_no, weights["mask"], "rel_attn"
        )
        attn_df = self.process_batch(
            weights["attn_weights"], event_no, weights["mask"], "attn"
        )

        # Concatenate the DataFrames for the current batch
        batch_df = pd.concat([rel_attn_df, attn_df])

        # Group by event_number and pivot to have a column for each matrix
        grouped_df = (
            batch_df.groupby(["event_number", "attn_type", "layer_no", "head_no"])[
                "matrix"
            ]
            .first()
            .unstack(level=["attn_type", "layer_no", "head_no"])
        )

        # Format the column names
        if isinstance(grouped_df.columns, pd.MultiIndex):
            grouped_df.columns = grouped_df.columns.map("{0[0]}_{0[1]}_{0[2]}".format)

        # Dynamically add columns to the SQLite table if they do not exist
        for column in grouped_df.columns:
            try:
                self.cursor.execute(
                    f"""ALTER TABLE attention_weights ADD COLUMN {column} BLOB"""
                )
                self.conn.commit()
            except sqlite3.OperationalError as e:
                # If the column already exists, an OperationalError will be raised, which we can ignore
                pass

        # Insert a row for each event_number into the SQLite database
        for event_number, row in grouped_df.iterrows():
            columns_str = ", ".join(row.index)
            placeholders = ", ".join("?" * len(row))
            self.cursor.execute(
                f"""INSERT OR IGNORE INTO attention_weights (event_number, {columns_str})
                                    VALUES (?, {placeholders})""",
                (event_number, *row.values),
            )
            self.conn.commit()

    def on_test_end(self, trainer, pl_module):
        # Close the SQLite database connection
        self.conn.close()

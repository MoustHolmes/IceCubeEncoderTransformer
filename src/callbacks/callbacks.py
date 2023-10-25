import pandas as pd
import torch
import numpy as np
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
        print("saving predictions to", self.save_path)
        df = pd.DataFrame(
            {
                "alpha": all_preds[:, 0].flatten(),
                "beta": all_preds[:, 1].flatten(),
                "targets": all_targets.flatten(),
                "event_no": all_event_nos.flatten(),
            }
        )

        df.to_csv(self.save_path, index=False)


class AttentionWeightsWriterCallback(Callback):
    def __init__(self, save_path):
        self.save_path = save_path

    def remove_padding(self, matrix, mask):
        # Find the indices of the non-padded elements
        non_padded_indices = np.where(mask != float("-inf"))[0]
        # Remove padding from the matrix
        return matrix[np.ix_(non_padded_indices, non_padded_indices)]

    def process_batch(self, attn_batch, batch_event_no, mask, attn_type):
        batch_size, n_layers, n_heads, max_seq_len, _ = attn_batch.shape
        for event_idx in range(batch_size):
            for layer_no in range(n_layers):
                for head_no in range(n_heads):
                    matrix = (
                        attn_batch[event_idx, layer_no, head_no]
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )  # Convert PyTorch tensor to NumPy Float64 array
                    matrix = self.remove_padding(matrix, mask[event_idx].cpu().numpy())
                    self.data.append(
                        {
                            "event_number": batch_event_no[event_idx].cpu().numpy(),
                            "attn_type": attn_type,
                            "layer_no": layer_no,
                            "head_no": head_no,
                            "matrix": matrix.tobytes(),
                        }
                    )

    def on_test_start(self, trainer, pl_module):
        self.data = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.process_batch(
            outputs["rel_attn"], outputs["event_no"], outputs["maks"], "rel_attn"
        )
        self.process_batch(
            outputs["attn"], outputs["event_no"], outputs["maks"], "attn"
        )

    def on_test_end(self, trainer, pl_module):
        df = pd.DataFrame(self.data)
        df.to_parquet(self.save_path)

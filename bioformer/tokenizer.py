import torch
from torch import nn

import numpy as np
import pandas as pd 
from typing import Dict, List, Tuple, Union

from vocab import Vocab

class Tokenizer(nn.Module):

    def __init__(self,
                 vocab: Vocab,
                 append_cls = True,
                 cls_token = "<cls>",
                 pad_token = "<pad>",
                 pad_value = -2,
                 include_zero_gene: bool = False,
                 ):
        super(Tokenizer, self).__init__()
        self.vocab = vocab
        self.append_cls = append_cls
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.include_zero_gene = include_zero_gene

    def tokenize_batch(
        self,
        data: np.ndarray,
        gene_ids: np.ndarray,
        ) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
        """
        Tokenize a batch of data. Returns a list of tuple (gene_id, count).

        Args:
            data (array-like): A batch of data, with shape (batch_size, n_features).
                n_features equals the number of all genes.
            gene_ids (array-like): A batch of gene ids, with shape (n_features,).
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            list: A list of tuple (gene_id, count) of non zero gene expressions.
        """
        if data.shape[1] != len(gene_ids):
            raise ValueError(
                f"Number of features in data ({data.shape[1]}) does not match "
                f"number of gene_ids ({len(gene_ids)})."
            )

        tokenized_data = []
        for i in range(len(data)):

            row = data[i]

            if self.include_zero_gene:
                values = row
                genes = gene_ids

            else:
                idx = np.nonzero(row)[0]
                values = row[idx]
                genes = gene_ids[idx]

            if self.append_cls:
                genes = np.insert(genes, 0, self.vocab[self.cls_token])
                values = np.insert(values, 0, 0)

            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()

            tokenized_data.append((genes, values))
        return tokenized_data

    def pad_batch(
        self,
        batch: List[Tuple],
        max_len: int,
        ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of data. Returns a list of Dict[gene_id, count].

        Args:
            batch (list): A list of tuple (gene_id, count).
            max_len (int): The maximum length of the batch.
            vocab (Vocab): The vocabulary containing the pad token.
            pad_token (str): The token to pad with.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of gene_id and count.
        """
        max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
        max_len = min(max_ori_len, max_len)

        pad_id = self.vocab[self.pad_token]

        gene_ids_list = []
        values_list = []
        

        for i in range(len(batch)):

            gene_ids, values = batch[i]

            if len(gene_ids) > max_len:

                if not self.append_cls:
                    idx = np.random.choice(len(gene_ids), max_len, replace=False)
                else:
                    idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
                gene_ids = gene_ids[idx]
                values = values[idx]

            if len(gene_ids) < max_len:
                gene_ids = torch.cat(
                    [
                        gene_ids,
                        torch.full(
                            (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                        ),
                    ]
                )
                values = torch.cat(
                    [
                        values,
                        torch.full((max_len - len(values),), self.pad_value, dtype=values.dtype),
                    ]
                )

            gene_ids_list.append(gene_ids)
            values_list.append(values)

        batch_padded = {
            "genes": torch.stack(gene_ids_list, dim=0),
            "values": torch.stack(values_list, dim=0),
        }

        return batch_padded

    def tokenize_and_pad_batch(
        self,
        data: np.ndarray,
        gene_ids: np.ndarray,
        max_len: int,
        ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
        """

        tokenized_data = self.tokenize_batch(
            data,
            gene_ids,
        )

        batch_padded = self.pad_batch(
            tokenized_data,
            max_len,
        )
        return batch_padded

def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_single_value: bool = False,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
    ) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like): A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio) if not mask_single_value else 1
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()

def retrieve_tfs(
    input_gene_ids,     
    input_values: Union[torch.Tensor, np.ndarray],  # masked
    tf,
    vocab: Vocab,
    mask_value: int = -1,
    randomize_c: bool = False,
    ) -> torch.Tensor:
    """
    Create conditions vector from tf_lookup, <msk> and gene_ids. 
    c[j] = -1 if j inhibits <msk>, 1 if stimulates, 0 otherwise

    Args:
        values (array-like): A batch of tokenized data with 1 entry masked, with shape (batch_size, n_features).
        gene_ids (array-like): The corresponding gene tokens ids.
        tf (pd.DataFrame): A lookup dataframe with columns [source, target, is_stimulation, is_inhibition]

    Returns:
        torch.Tensor: A tensor of conditions vectors.
    """

    itos = vocab.get_itos()
    conditions = np.zeros_like(input_values)   # array
    if randomize_c:     
        conditions[:] = np.random.choice([-1,0,1], size = conditions.shape)
        print('Random C')

    else:
        count_targets, count_sources = 0, 0
        for i in range(len(conditions)):
            
            # Current gene ids
            gene_ids = input_gene_ids[i].tolist()
            
            # convert back to names
            gene_names = [itos[g] for g in gene_ids]
            
            # retrieve position of masked gene (idx == -1)
            masked_gene_position = torch.argwhere(input_values[i].eq(mask_value)).item()
            
            # retrieve name of masked gene
            target = gene_names[masked_gene_position]   # retrieve name of masked gene
            
            # restrict tf dataset to correct target
            subset_tf = tf[tf.target == target]

            # if correct target is present
            if len(subset_tf) > 0:
                count_targets += 1
                
                # subset to available sources
                subset_tf = subset_tf[subset_tf.source.isin(gene_names)]
                
                # and if among the genes there is a known source
                if len(subset_tf) > 0:
                    count_sources += 1
                    for j in range(len(subset_tf)):
                        source_position = gene_names.index(subset_tf.iloc[j].source)

                        # TODO: try categorical
                        if subset_tf.iloc[j].is_stimulation == 1:
                            conditions[i, source_position] = 1
                        elif subset_tf.iloc[j].is_inhibition == 1:
                            conditions[i, source_position] = -1

        print('Available Targets: ', count_targets)
        print('With at least one available Source: ', count_sources)
    return conditions



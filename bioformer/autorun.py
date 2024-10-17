import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import scanpy as sc
import time
import copy
from typing import List, Tuple, Dict, Union, Optional
from sklearn.model_selection import train_test_split

from utils import set_seed, AttrDict
from vocab import Vocab
from preprocess import Preprocessor, get_interactions, get_z
from tokenizer import tokenize_and_pad_batch, random_mask_value
from model import TransformerModel, BioFormerModel
from loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli

config = AttrDict(json.load(open('config.json')))
print(config)

if config.seed:
    set_seed(config.seed)

if config.wandb:
    import wandb
    wandb.login()
    run = wandb.init(
        project='BioFormer',
        config = config,
        name = config.run_name if config.run_name else None
    )

# Pre-processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_value = -1 # in the value vector corresponding to msk token (!= msk token index in vocab)
pad_value = -2  # in the value vector corresponding to pad token (!= pad token index in vocab)
n_input_bins = config.n_bins
include_zero_gene = config.include_zero_gene
n_hvg = config.n_hvg
max_seq_len = n_hvg + 1

# Import data
path_to_transcriptional_interactions = '../data/transcriptional_interactions.csv'
dataset_name = config.dataset_name

if dataset_name == 'BREAST_25K':
    adata = sc.read_h5ad('../data/breast_25k.h5ad')
    data_is_raw = True

elif dataset_name == 'BREAST_12K':
    adata = sc.read_h5ad('../data/breast_12k.h5ad')
    data_is_raw = True

elif dataset_name == 'DERMAL_100K':
    adata = sc.read_h5ad('../data/dermal_100k.h5ad')
    adata.var["gene_name"] = adata.var.feature_name.tolist()
    data_is_raw = True

elif dataset_name == 'HYPOXIA_9K':
    adata = sc.read_h5ad('../data/scsHypoxiaTimeSub.h5ad')
    adata.X = adata.layers['raw_count']
    adata.var['gene_name'] = adata.var.index.tolist()
    data_is_raw = True

print(dataset_name)
print(adata)

# Pre-process adata
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=config.n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

preprocessor(adata, batch_key=None)

input_layer_key = "X_binned"
all_counts = adata.layers[input_layer_key].toarray()
genes = adata.var["gene_name"].tolist()

# Vocab
vocab = Vocab(genes + special_tokens)
vocab.set_default_index(vocab["<pad>"]) # index to return if token not found in vocab
gene_ids = np.array(vocab(genes), dtype=int)
print(f'Vocab of size: {len(vocab)} --> {len(genes)} genes, {len(special_tokens)} special tokens {special_tokens}')

tokenized = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)

print(f"Tot samples: {tokenized['genes'].shape[0]}")
print(f"Input length: {tokenized['genes'].shape[1]}")

def prepare_data():
    
    masked_values = random_mask_value(
        tokenized["values"],
        mask_value=mask_value,
        pad_value=pad_value,
        mask_single_value = config.mask_single_value
    )

    print(f"random masking at epoch {epoch}, ratio of masked values: {(masked_values == mask_value).sum() / (masked_values - pad_value).count_nonzero():.4f}")

    B, r = masked_values.shape
    
    if config.init_z:
        tf = get_interactions(genes,path_to_transcriptional_interactions)
    z_train = get_z(tokenized["genes"], tf, vocab.itos) if config.init_z else torch.zeros((B, r, r))    # [B, r, r]

    data_pt = {
        "gene_ids": tokenized["genes"],           # [B, r]
        "values": masked_values,                  # [B, r]
        "target_values": tokenized["values"],     # [B, r]
        "z": z_train
    }

    dataset = SeqDataset(data_pt)
    return dataset

# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

# data_loader
def prepare_dataloader(
    # data_pt: Dict[str, torch.Tensor],
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    
    # dataset = SeqDataset(data_pt)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntoken = len(vocab)  # size of vocabulary
if config.model == "scGPT":
    model = TransformerModel(
        ntoken=ntoken,
        d_model=config.d_model,
        nhead=config.nhead,
        d_hid=config.d_model*4,
        nlayers=config.nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
    ) 
elif config.model == "BioFormer":
    model = BioFormerModel(
        ntoken=ntoken,
        d_model=config.d_model,
        nhead=config.nhead,
        nlayers=config.nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        do_pair_bias=config.do_pair_bias,
        do_opm=config.do_opm,
    ) 

model.to(device)
model = torch.nn.DataParallel(model)

n_params = sum(p.numel() for p in model.parameters())
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

if config.wandb:
    wandb.config.update({"Model Parameters": n_params})

print(f'''
device: {device} | model: {config.model} | d_model: {config.d_model} | nhead: {config.nhead} | nlayers: {config.nlayers} | tot. params: {n_params/1e6:.2f}M | model size: {model_size_bytes/1e6:.2f}MB
''')

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_mre = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        
        if config.model == "BioFormer":
            z = batch_data['z'].to(device)
            # B, r = input_values.shape
            # z = torch.randn((B, r, r)).to(device)
        

        # ---------- FORWARD PASS -------------------
        with torch.cuda.amp.autocast(enabled=config.amp):
            
            if config.model == "scGPT":
                output_dict = model(input_gene_ids, input_values)
            elif config.model == "BioFormer":
                output_dict = model(input_gene_ids, input_values, z)
            
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
            
            metrics_to_log = {"train/mse": loss_mse.item()}
            
            if config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(output_dict["mlm_zero_probs"], target_values, masked_positions)
                loss += loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            
            if config.GEPC:
                loss_gepc = criterion(output_dict["mvc_output"], target_values, masked_positions)
                loss += loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            
            if config.GEPC and config.explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(output_dict["mvc_zero_probs"], target_values, masked_positions)
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_gepc_zero_log_prob.item()})

        # ---------- BACKWARD PASS ------------------
        model.zero_grad()
        scaler.scale(loss).backward()   # training via the aggregated loss
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        # -------------------------------------------
        
        if config.wandb:
            wandb.log(metrics_to_log)

        # Compute MRE for validation
        with torch.no_grad():
            mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)

        total_loss += loss.item()                               # sum of all losses
        total_mse += loss_mse.item()                            # MSE alone
        total_gepc += loss_gepc.item() if config.GEPC else 0.0  # MSE from GEPC alone
        total_mre += mre.item()                                 # MRE alone
        
        # Avg of loss across all log_interval batches (i.e., log_interval = 10, avg loss every 10 batches)
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_mre = total_mre / log_interval
            
            print(f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | train/loss {cur_loss:5.2f} | train/mse {cur_mse:5.2f} |" + (f"train/gepc {cur_gepc:5.2f} |" if config.GEPC else "") + f"train/mre {cur_mre:5.2f} |" )
            
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_mre = 0
            start_time = time.time()

def define_wandb_metrics():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_mre = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            if config.model == "BioFormer":
                z = batch_data['z'].to(device)
                # B, r = input_values.shape
                # z = torch.randn((B, r, r)).to(device)

            with torch.cuda.amp.autocast(enabled=config.amp):
                
                if config.model == "scGPT":
                    output_dict = model(input_gene_ids, input_values)
                elif config.model == "BioFormer":
                    output_dict = model(input_gene_ids, input_values, z)
                
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)

            total_loss += loss.item() * len(input_gene_ids)
            total_mre += masked_relative_error(output_values, target_values, masked_positions).item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    if config.wandb:
        wandb.log({ 
            "valid/mse": total_loss / total_num,
            "valid/mre": total_mre / total_num,
            "epoch": epoch
            })

    return total_loss / total_num, total_mre / total_num

best_val_loss = float("inf")
best_model = None

if config.wandb:
    define_wandb_metrics()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    
    dataset = prepare_data()
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # TRAINING      --> over all batches in the train_loader
    if config.do_train:
        train(model, loader=train_loader)

    # VALIDATION    --> avg loss across all batches in valid_loader
    val_loss, val_mre = evaluate(model, loader=valid_loader)
    
    
    # Some epoch-related stats
    elapsed = time.time() - epoch_start_time
    print("-" * 89)
    print(f"| end of epoch {epoch:3d} | runtime: {elapsed:5.2f}s | valid/mse {val_loss:5.4f} | valid/mre {val_mre:5.4f}")
    print("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        print(f"Best model with valid/mse {best_val_loss:5.4f}")

    scheduler.step()

if config.wandb:
    run.finish()
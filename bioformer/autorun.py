import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import gc
import json
import numpy as np
import scanpy as sc
import time
import copy
from scipy.sparse import issparse

from utils import set_seed, AttrDict
from vocab import Vocab
from preprocess import Preprocessor, get_interactions, get_z
from tokenizer import Tokenizer
from model import TransformerModel, BioFormerModel
from loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli

# import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

config = AttrDict(json.load(open('./config.json')))
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

# Import data
path_to_transcriptional_interactions = '../data/transcriptional_interactions.csv'
dataset_name = config.dataset_name

if dataset_name == 'BREAST_25K':
    adata = sc.read_h5ad('../data/breast_25k.h5ad')
    data_is_raw = True

elif dataset_name == 'BREAST_2M':
    adata = sc.read_h5ad('../data/breast_2M.h5ad')
    data_is_raw = True

elif dataset_name == 'BREAST_800K':
    adata = sc.read_h5ad('../data/breast_800k.h5ad')
    data_is_raw = True

elif dataset_name == 'HYPOXIA_9K':
    adata = sc.read_h5ad('../data/scsHypoxiaTimeSub.h5ad')[:100]
    adata.X = adata.layers['raw_count']
    adata.var['gene_name'] = adata.var.index.tolist()
    data_is_raw = True

print(dataset_name)
print(adata)

# Pre-process RNA-seq data
preprocessor = Preprocessor(use_key="X",  # the key in adata.layers to use as raw data
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

# Vocab
genes = adata.var["gene_name"].tolist()
vocab = Vocab(genes)
vocab.set_default_index(vocab["<pad>"]) # index to return if token not found in vocab
print(f'Init vocab of size {len(vocab)} with {config.n_hvg} unique genes...')
print(f"CLS in vocab: {vocab.stoi['<cls>']}")

# Tokenize & Pad
tokenizer = Tokenizer(vocab = vocab,
                      append_cls = False,
                      pad_token = "<pad>",
                      pad_value = -2,
                      include_zero_gene= config.include_zero_gene, 
                      mask_value = -1,
                      )
tokenized = tokenizer.tokenize_and_pad_batch(adata.layers["X_binned"].toarray() if issparse(adata.layers["X_binned"]) else adata.layers["X_binned"],
                                             np.array(vocab(genes), dtype=int),
                                             max_len=config.n_hvg,
                                             )
del adata   # free up memory
print(f"Tot samples: {tokenized['genes'].shape[0]}")
print(f"Input length: {tokenized['genes'].shape[1]}")

# Instantiate model
if config.model == "scGPT":
    model = TransformerModel(ntoken=len(vocab),
                             d_model=config.d_model,
                             nhead=config.nhead,
                             nlayers=config.nlayers,
                             pad_id = vocab.stoi['<pad>'],
                             explicit_zero_prob=config.explicit_zero_prob
                             ) 
elif config.model == "BioFormer":
    model = BioFormerModel(ntoken=len(vocab),
                           d_model=config.d_model,
                           d_z = config.d_z,
                           d_opm = config.d_opm,
                           nhead=config.nhead,
                           nlayers=config.nlayers,
                           do_pair_bias=config.do_pair_bias,
                           do_opm=config.do_opm,
                           pad_id = vocab.stoi['<pad>'],
                           explicit_zero_prob=config.explicit_zero_prob
                           ) 
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# model = torch.nn.DataParallel(model)

# Parameters count
n_params = sum(p.numel() for p in model.parameters())
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f'''device: {device} | model: {config.model} | d_model: {config.d_model} | nhead: {config.nhead} | nlayers: {config.nlayers} | tot. params: {n_params/1e6:.2f}M | model size: {model_size_bytes/1e6:.2f}MB''')
if config.wandb:
    wandb.config.update({"Model Parameters": n_params})

# Max memory required per intermediate step:
if config.model == 'BioFormer':
        # no. elements in the [B, r, r, c, c] opm intermediate matrix
        tmp = config.batch_size * (config.n_hvg + 1) ** 2 * config.d_opm ** 2
        
        # number of bytes required (using np.float32)
        tmp = tmp * 4

        print(f'memory required for opm: {tmp/1e6 :,.2f}MB')
        

# RNA-seq Dataset
class SeqDataset(Dataset):
    """
    Create RNA-seq dataset from vocabulary with keys ['gene_ids', 'valeus', 'target_vaules', 'interactions'].
    """
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# Mask and get interactions
def prepare_data():
    """
    1. Random mask the data
    2. Get the interaction matrix z
    3. Convert to torch.Dataset.
    
    """

    print(f"Random masking at epoch {epoch}...")
    masked_values = tokenizer.random_mask_value(tokenized["values"],
                                                mask_single_value = False,
                                                mask_ratio = 0.15,
                                                )
    
    B, r = masked_values.shape
    if config.init_z:
        tf = get_interactions(genes, path_to_transcriptional_interactions)
    interactions = get_z(tokenized["genes"], tf, vocab.itos) if config.init_z else torch.zeros((B, r, r))    # [B, r, r]

    data_pt = {
        "gene_ids": tokenized["genes"],           # [B, r]
        "values": masked_values,                  # [B, r]
        "target_values": tokenized["values"],     # [B, r]
        "interactions": interactions              # [B, r, r]
    }

    print(f'Splitting data into train and valid at epoch {epoch}...')
    return SeqDataset(data_pt)

# --------------------------------------------------------------------------- #
# --------------------------- TRAINING LOOP --------------------------------- #
# --------------------------------------------------------------------------- #

criterion = masked_mse_loss
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=config.lr,
                              eps=1e-4 if config.amp else 1e-8
                              )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                             1,
#                                             gamma=config.schedule_ratio
#                                             )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       patience=2,  
                                                       threshold=1                                                                                          
                                                       )
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

best_val_loss = float("inf")
best_model = None

if config.wandb:
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")

for epoch in range(1, config.epochs + 1):

    print("-" * 89)
    print(f"| Epoch {epoch} of {config.epochs}...")
    print("-" * 89)
    print(f"Allocated memory at epoch {epoch}: {torch.cuda.memory_allocated() / 1e6:,.2f} MB")
    print(f"Cached memory at epoch {epoch}: {torch.cuda.memory_reserved() / 1e6:,.2f} MB")

    epoch_start_time = time.time()
    
    if epoch > 1:
        del dataset, train_dataset, valid_dataset, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

    dataset = prepare_data()

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # -------------------------------- TRAINING ----------------------------------- #
    if config.do_train:
        model.train()

        loader = train_loader

        total_loss = 0.0
        total_mse = 0.0
        total_gepc = 0.0
        total_mre = 0.0
        log_interval = config.log_interval
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            
            if config.model == "BioFormer":
                z = batch_data['interactions'].to(device)

            # ---------- forward -------------------
            with torch.cuda.amp.autocast(enabled=config.amp):
                
                if config.model == "scGPT":
                    output_dict = model(input_gene_ids, input_values)
                elif config.model == "BioFormer":
                    output_dict = model(input_gene_ids, input_values, z)
                
                masked_positions = input_values.eq(-1)          # default value for the mask position
                loss = loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
                
                metrics_to_log = {"train/mse": loss_mse.item()}
                
                if config.explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(output_dict["mlm_zero_probs"], target_values, masked_positions)
                    loss += loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                
            # -------------- backward ------------------
            model.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            # --------------- logs & stats ---------------------
            if config.wandb:
                wandb.log(metrics_to_log)

            with torch.no_grad():
                mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)

            total_loss += loss.item()                               # sum of all losses
            total_mse += loss_mse.item()                            # MSE alone
            total_mre += mre.item()                                 # MRE alone
            
            # For logging purposes, aggregate loss across log_interval batches 
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_mre = total_mre / log_interval
                
                print(f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | train/loss {cur_loss:5.2f} | train/mse {cur_mse:5.2f} |" + f"train/mre {cur_mre:5.2f} |" )
                
                total_loss = 0
                total_mse = 0
                total_mre = 0
                start_time = time.time()

    # -------------------------------- VALIDATION ----------------------------------- #
    model.eval()
    
    loader = valid_loader
    
    total_loss = 0.0
    total_mre = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            if config.model == "BioFormer":
                interactions = batch_data['interactions'].to(device)

            with torch.cuda.amp.autocast(enabled=config.amp):
                
                if config.model == "scGPT":
                    output_dict = model(input_gene_ids, input_values)
                elif config.model == "BioFormer":
                    output_dict = model(input_gene_ids, input_values, interactions)
                
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(-1)
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

    val_loss = total_loss / total_num
    val_mre = total_mre / total_num
    
    # -------------------------------- EPOCH-RELATED STATS ----------------------------------- #
    elapsed = time.time() - epoch_start_time
    print("-" * 89)
    print(f"| end of epoch {epoch:3d} | runtime: {elapsed:5.2f}s | valid/mse {val_loss:5.4f} | valid/mre {val_mre:5.4f}")
    print("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        print(f"New best model found at epoch {epoch} with valid/mse {best_val_loss:5.4f}")

    # Update lr
    scheduler.step(val_loss)

# --------------------------------- END OF TRAINING LOOP -------------------------------------- #

# --------------------------------- final house-keeping --------------------------------------- #
if config.save_model:
    
    if config.save_model[-1] != "/":
        config.save_model += "/"
    
    dir = f"{config.save_model}/{config.run_name}_{time.time():.0f}"
    print(f"Saving model to {dir}...")

    # Save Model
    torch.save(best_model.state_dict(), f"{dir}.pt")
    
    # Save Args
    with open(f"{dir}.json", "w") as f:
        json.dump(config, f)

if config.wandb:
    run.finish()
import random
import numpy as np
import torch
import pandas as pd 

# Dict with attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_interactions(
        genes: list=None,
        path_to_file: str=None
    )->pd.DataFrame:
    """
    
    Given a list of genes, returns the available interactions.
    
    *args:
        genes: 
            <list> of <str> containing gene names.

    
    Returns:
        <Pandas.DataFrame> with columns ['source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']
    
    """
    df = pd.read_csv(path_to_file, low_memory=False, index_col = 0)
    print(f'Interactions before filtering: {df.shape[0]:,}')
    df = df.loc[df.source_genesymbol.isin(genes)]
    df = df.loc[df.target_genesymbol.isin(genes)]
    print(f'Interactions after filtering: {df.shape[0]}')
    
    return df[['source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']]

# Given list of genes, generate the pair-representation matrix filled with interations
def get_z(
        g: torch.Tensor = None,
        interactions: pd.DataFrame = None,
        itos: dict = None
    )->torch.Tensor:
    """
    *args:
        g:
            [B, r]:
                   vectors of gene tokens (gene_ids)
        interactions:
            pandas.DataFrame with columns ['source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']
    
    Returns:
        z:
            [B, r, r] <pyTorch.Tensor>
    """
    # TODO
    # import transcriptional_interactions and filter vocabulary
    # (maybe a cache to avoid importing and checking vocabulary every time?)
    # init (B, r, r) z tensor
    # loop through all genes and fill z
    B, r = g.shape
    counter = torch.zeros(B)
    z = torch.zeros((B, r, r))
    for b in range(B):  # batches
        if b % 50 == 0: print(f'Parsing batch {b:04}/{B}')
        for i in range(r):
            # if interactions.source_genesymbol.str.contains(itos[g[b, i].item()]).any():
            #     counter[b] += 1
                for j in range(0, i+1):
                    
                    # i source, j target
                    interaction_type = interactions.loc[(interactions.source_genesymbol == itos[g[b, i].item()]) & (interactions.target_genesymbol == itos[g[b, j].item()])]
                    if interaction_type.shape[0] == 1:
                        z[b, i, j] +=  interaction_type.is_stimulation.item()
                        z[b, i, j] -=  interaction_type.is_inhibition.item()
                    
                    # j source, i target
                    interaction_type = interactions.loc[(interactions.source_genesymbol == itos[g[b, j].item()]) & (interactions.target_genesymbol == itos[g[b, i].item()])]
                    if interaction_type.shape[0] == 1:
                        z[b, j, i] +=  interaction_type.is_stimulation.item()
                        z[b, j, i] -=  interaction_type.is_inhibition.item()
    print(f'Avg sources found per batch: {counter.mean():.2f}')
    return z


import torch 
import torch.nn as nn

class OuterProductMean:
    """
    Input:
        RNA-seq:
            (r, c) tensor of sequence representation
    
    Output:
        Pair-wise updates:
            (r, r, c) biases to update pair-wise representation

    """
    def __init__(self) -> None:
        pass

class RowAttentionWithPairBias:
    """
    Input:
        RNA-seq:
            (r, c)
        
        TF:
            (r, r, c)
    
    Output:
        Attention:
            (r, c)

    Compute attn between any two vectors i,j in the input RNA-seq using TF[i,j] as pair bias in the attn computation
    """
    def __init__(self) -> None:
        pass


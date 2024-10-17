import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# B = batch_size
# r = number of residues in the sequence
# c = input embedding dimensions

class BioFormerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        # d_hid: int,
        nlayers: int,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        do_opm: bool = True,
        do_pair_bias: bool = True,
        # pad_value: int = 0,
        ):
        super().__init__()
        self.d_model = d_model

        self.emb_g = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.emb_x = ContinuousValueEncoder(d_model, dropout)
        self.emb_z = ContinuousValueEncoder(d_model, dropout)

        self.bioformer = BioFormerStack(
                            c_m=d_model,
                            c_z=d_model,
                            c_hidden=d_model,
                            no_heads=nhead,
                            no_blocks=nlayers,
                            do_opm=do_opm,
                            do_pair_bias=do_pair_bias,
                            )
        # encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = ExprDecoder(d_model, explicit_zero_prob = True)
        self.mvc_decoder = MVCDecoder(d_model, explicit_zero_prob = True)

        self.cell_emb_style = "cls"

    def _encode(
        self,
        g: Tensor,
        x: Tensor,
        z: Optional[Tensor]=None,
        ) -> Tensor:
        
        g = self.emb_g(g)       # (batch, seq_len, embsize)
        x = self.emb_x(x)       # (batch, seq_len, embsize)
        z = self.emb_z(z)    # (batch, seq_len, seq_len, emb_size)
        
        self.cur_gene_token_embs = x
        
        total_embs = g + x

        # output = self.transformer_encoder(total_embs)
        output, z = self.bioformer(total_embs, z)
        return output  # (batch, seq_len, embsize)

    def forward(
        self,
        g: Tensor,
        x: Tensor,
        z: Optional[Tensor]=None,
        ):
            
        transformer_output = self._encode(g, x, z)

        output = {}
        

        # Masked Value Prediction
        mlm_output = self.decoder(transformer_output)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        output["mlm_zero_probs"] = mlm_output["zero_probs"]
        
        # Masked Value Prediction (<cls> only)
        # cell_emb = self._get_cell_emb_from_layer(transformer_output)
        # mvc_output = self.mvc_decoder(cell_emb)
        # output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
        # output["mvc_zero_probs"] = mvc_output["zero_probs"]
        # output["cell_emb"] = cell_emb

        return output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb
    
class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        # pad_value: int = 0,
        ):
        super().__init__()
        self.d_model = d_model

        self.emb_g = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.emb_x = ContinuousValueEncoder(d_model, dropout)
       
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = ExprDecoder(d_model, explicit_zero_prob = True)
        self.mvc_decoder = MVCDecoder(d_model, explicit_zero_prob = True)

        self.cell_emb_style = "cls"

    def _encode(
        self,
        g: Tensor,
        x: Tensor,
        ) -> Tensor:

        g = self.emb_g(g)       # (batch, seq_len, embsize)
        x = self.emb_x(x)       # (batch, seq_len, embsize)
        
        self.cur_gene_token_embs = x
        
        total_embs = g + x

        output = self.transformer_encoder(total_embs)
        return output  # (batch, seq_len, embsize)

    def forward(
        self,
        g: Tensor,
        x: Tensor,
        ):
        transformer_output = self._encode(g, x)

        output = {}
        

        # Masked Value Prediction
        mlm_output = self.decoder(transformer_output)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        output["mlm_zero_probs"] = mlm_output["zero_probs"]
        
        # Masked Value Prediction (<cls> only)
        # cell_emb = self._get_cell_emb_from_layer(transformer_output)
        # mvc_output = self.mvc_decoder(cell_emb)
        # output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
        # output["mvc_zero_probs"] = mvc_output["zero_probs"]
        # output["cell_emb"] = cell_emb

        return output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

# scGPT Modules
class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)

class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        

# BioFormer Modules
class OuterProductMean(nn.Module):
    """
    
    Compute outer product mean of a given tensor.
    
    """
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                Input embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        # [*, r, r, c, c]
        outer = torch.einsum("...ab,...cd->...acbd", a, b)

        # [*, r, r, c * c]
        outer = outer.reshape(outer.shape[:-2] + (-1,)) # flatten last two dim

        # [*, r, r, c_z]
        outer = self.linear_out(outer)

        return outer

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [B, r, c] Input embedding

        Returns:
            [B, r, r, c_z] pair embedding update
        """

        # [*, r, c_m]
        ln = self.layer_norm(m)

        a = self.linear_1(ln)         
        b = self.linear_2(ln) 

        del ln

        outer = self._opm(a, b)

        return outer

class RowAttentionWithPairBias(nn.Module):
    """

    Compute a given sequence self-attention using provided biases.
    
    """
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=True,
        c_z=None,
        gating=True,
    )-> None:
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            gating:
                Whether to use gated attention or not
        """
        super(RowAttentionWithPairBias, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.gating = gating

        # Initial Normalization
        self.layer_norm_m = nn.LayerNorm(self.c_in)

        # Bias
        self.layer_norm_z = nn.LayerNorm(self.c_z) if self.pair_bias else None
        self.linear_z = nn.Linear(self.c_z, self.no_heads, bias=False) if self.pair_bias else None

        # Queries, Keys, Values
        self.linear_q = nn.Linear(self.c_in, self.c_hidden, bias=False)
        self.linear_k = nn.Linear(self.c_in, self.c_hidden, bias=False)
        self.linear_v = nn.Linear(self.c_in, self.c_hidden, bias=False)
        
        # Gating
        self.linear_g = nn.Linear(self.c_in, self.c_hidden) if self.gating else None
        
        # Final projection
        self.linear_o = nn.Linear(self.c_hidden, self.c_in)

    def forward(self,
            m: torch.Tensor,
            z: Optional[torch.Tensor] = None,
    )->torch.Tensor:
        """
        Args:
            m:
                [B, r, c] Input embedding
            z:
                [B, r, r, c_z] pair embedding. Required only if
                pair_bias is True
        """
        if z is None and self.pair_bias is True:
            raise Warning("z required when pair bias is true")
        
        # [B, r, c]
        m = self.layer_norm_m(m)

        # [B, r, c_hid * H]
        q = self.linear_q(m)
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, r, H, c_hid], where H = no_heads
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, r, c_hid]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)   # scaled attn

        # [*, H, c_hid, r]
        k = torch.permute(k, (0, 1, 3, 2))  # flip last two dims for matmul

        # [*, H, r, r]
        a = torch.matmul(q, k)

        if self.pair_bias:
            # [B, r, r, c_z]
            z = self.layer_norm_z(z)
            # [B, r, r, H]
            z = self.linear_z(z)
            # [B, H, r, r]
            z = torch.permute(z, (0, 3, 1, 2))
            # [B, H, r, r]
            a = a + z
            # print('Pair Bias')

        # [B, H, r, r]
        a = nn.functional.softmax(a, -1)

        # [B, H, r, c_hid]
        a = torch.matmul(a, v)

        # [B, r, H, c_hid]
        a = a.transpose(-2, -3)
        
        if self.gating:
            # [B, r, c_hid * H]
            g = nn.functional.sigmoid(self.linear_g(m))
            # [B, r, H, c_hid]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            # [B, r, H, c_hid]
            a = a * g

        # [B, r, H * c_hid]
        a = a.reshape(a.shape[:-2] + (-1,))     # flatten H dim

        # [B, r, c_m]
        a = self.linear_o(a)
        
        return a

class Transition(nn.Module):
    """
    Feed-forward network.
    """
    def __init__(self, c_m, n=4):
        """
        Args:
            c_m:
                Input channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension (default is 4).
        """
        super(Transition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = nn.LayerNorm(self.c_m)
        self.linear_1 = nn.Linear(self.c_m, self.n * self.c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.n * self.c_m, self.c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [B, r, r, c] Input 

        Returns:
            m:
                [B, r, r, c] Update
        """
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)

        return m
    
class BioFormerBlock(nn.Module):
    """
    
    A single BioFormer block.
    It performs gated-self attention with pair bias, transition and outer-product mean sequentially via residual connections.

    """
    def __init__(
                self,
                c_m,
                c_z,
                c_hidden,
                no_heads,
                do_opm,
                do_pair_bias,
                ):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            do_opm:
                Whether to compute the outer product mean update
            do_pair_bias:
                Whether to use the pair bias in the self-attention layer
        """
        super(BioFormerBlock, self).__init__()
        
        self.do_opm, self.do_pair_bias = do_opm, do_pair_bias
        
        self.attn = RowAttentionWithPairBias(
                                        c_in=c_m,
                                        c_hidden=c_hidden,
                                        no_heads=no_heads,
                                        pair_bias=self.do_pair_bias,
                                        c_z=c_z, 
                                        gating=False
                                        )
        self.trans = Transition(
                            c_m=c_m,
                            n=4
                            )
        self.opm = OuterProductMean(
                                c_m=c_m,
                                c_z=c_z,
                                c_hidden=c_hidden
                                ) if self.do_opm else None

    def forward(self,m, z):
        """
        Args:
            m:
                [B, r, c_m] RNA-seq input
            z:
                [B, r, r, c_z] pair representation
        Returns:
            m:
                [B, r, c_m] updated RNA-seqc
            z:
                [B, r, r, c_z] updated pair representation
        """
        m = m + self.attn(m, z)
        m = m + self.trans(m)
        
        if self.do_opm:
            z = z + self.opm(m)
            # print('opm update')

        return m, z
           
class BioFormerStack(nn.Module):
    """
    
    A series of BioFormer blocks.

    """
    def __init__(self,
            c_m,
            c_z,
            c_hidden,
            no_heads,
            no_blocks,
            do_opm,
            do_pair_bias,
            ):
        super(BioFormerStack, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = BioFormerBlock(
                                c_m=c_m,
                                c_z=c_z,
                                c_hidden=c_hidden,
                                no_heads=no_heads,
                                do_opm=do_opm,
                                do_pair_bias=do_pair_bias
                                )
            self.blocks.append(block)

    def forward(self, m, z):

        for b in self.blocks:
            m, z = b(m, z)
        
        return m, z
import torch
import torch.nn as nn

from typing import Dict, List, Optional

class Vocab(nn.Module):
    """
    
    Creates a vocab object which maps tokens to indices.
    
    """

    def __init__(self,
                 tokens,
                 default_index: Optional[int] = None
                 ) -> None:
        super(Vocab, self).__init__()
        
        self.stoi = {s:i for i, s in enumerate(tokens)}
        self.itos = {i:s for i, s in enumerate(tokens)}
        self.default_index = default_index

    @property
    def is_jitable(self):
        return isinstance(self.stoi, dict) and isinstance(self.itos, list)

    @torch.jit.export
    def forward(self, tokens: List[str]) -> List[int]:
        r"""Calls the `lookup_indices` method"""
        return self.lookup_indices(tokens)

    @torch.jit.export
    def __len__(self) -> int:
        r"""Returns the length of the vocab."""
        return len(self.itos)

    @torch.jit.export
    def __contains__(self, token: str) -> bool:
        r"""Checks if the token is in the vocab."""
        return token in self.stoi

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""Looks up the index of a token."""
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError(f"Token '{token}' not found in vocab and no default index set.")

    @torch.jit.export
    def set_default_index(self, index: Optional[int]) -> None:
        r"""Sets the default index."""
        self.default_index = index

    @torch.jit.export
    def get_default_index(self) -> Optional[int]:
        r"""Gets the default index."""
        return self.default_index

    @torch.jit.export
    def insert_token(self, token: str, index: int) -> None:
        r"""Inserts a token at a specific index."""
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocab.")
        if not 0 <= index <= len(self.itos):
            raise RuntimeError(f"Index '{index}' out of range [0, {len(self.itos)}].")
        self.stoi[token] = index
        self.itos.insert(index, token)
        # Update subsequent indices in stoi
        for i in range(index + 1, len(self.itos)):
            self.stoi[self.itos[i]] = i

    @torch.jit.export
    def append_token(self, token: str) -> None:
        r"""Appends a token to the vocab."""
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocab.")
        index = len(self.itos)
        self.stoi[token] = index
        self.itos.append(token)

    @torch.jit.export
    def lookup_token(self, index: int) -> str:
        r"""Looks up the token for a specific index."""
        if not 0 <= index < len(self.itos):
            raise RuntimeError(f"Index '{index}' out of range [0, {len(self.itos)}].")
        return self.itos[index]

    @torch.jit.export
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""Looks up tokens for a list of indices."""
        return [self.lookup_token(index) for index in indices]

    @torch.jit.export
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""Looks up indices for a list of tokens."""
        return [self[token] for token in tokens]

    @torch.jit.export
    def get_stoi(self) -> Dict[str, int]:
        r"""Gets the string-to-index dictionary."""
        return self.stoi

    @torch.jit.export
    def get_itos(self) -> List[str]:
        r"""Gets the index-to-string list."""
        return self.itos

    def __prepare_scriptable__(self):
        r"""Return a JITable Vocab."""
        if not self.is_jitable:
            raise RuntimeError("This Vocab instance cannot be JITable.")
        return self
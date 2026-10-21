"""A DynamicCache whose first write per layer stores the tensor instead of copying it.

transformers' DynamicLayer.update always runs torch.cat([empty, k]) on the
first call, a full copy of every layer's K and V (two of the four cat kernels
per layer in a prefill; ~5% of a 7k-token prefill on a b200, in both arms of
bench_ttft).  A serving stack writes K/V once.  Later updates (decode) still
concatenate, so the cache behaves exactly as DynamicCache from the second write
on, and everything else (get_seq_length, crop, reorder, the gate's in-place
pruning) is inherited.  Only plain full-attention layers are swapped; sliding
or linear-attention layers keep their own classes.
"""

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class _StoreFirstLayer(DynamicLayer):
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.keys.numel() == 0:
            self.keys, self.values = key_states, value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values


class PrefillCache(DynamicCache):
    def __init__(self, config):
        super().__init__(config=config)
        # Swap the class in place rather than constructing replacements:
        # DynamicLayer's __init__ signature differs across transformers 5.x,
        # and _StoreFirstLayer adds no state of its own.
        for layer in self.layers:
            if type(layer) is DynamicLayer:
                layer.__class__ = _StoreFirstLayer

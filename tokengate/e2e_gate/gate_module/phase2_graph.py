"""The pruned prefill's layer loop as one CUDA graph per (batch, padded length).

Phase 2 -- every layer above the pruning layer, run on the kept tokens -- is
launch-bound at the K values the benchmark plots: ~1k tokens through 26
layers cost a flat 9 ms of host time however the layers are compiled, and
inductor's per-layer CUDA graphs still left 4-6 ms of Python between them.
Here the whole loop is captured once per shape with torch.cuda.CUDAGraph and
replayed: the host issues one launch.

Exactness: the captured work is the same kernels on the same values; the
inputs are copied into the graph's static buffers, the outputs are the
graph's own tensors.  The caller pads the pruned length to a bucket
(set_phase2_bucket) so the set of shapes is small.

Contract for callers (prefill_batch_for_decode):
  - the layers must already be compiled in inductor's default mode (not
    reduce-overhead: cudagraph trees cannot be captured inside another graph);
  - anything the layers read as module attributes (the gate bias slice, the
    ssmax ratio, the attention scale) must live at a fixed address for the life
    of the process (Qwen3FlexAttention._static with its fixed-size buffers) and
    be written before replay;
  - the returned K/V tensors are the graph's: the next replay of the same
    shape overwrites them, so a caller that decodes from the cache must be
    done with it (or clone it) before the next prefill of that shape.
"""

import gc
import os

import torch
from transformers.cache_utils import DynamicLayer

from tokengate.e2e_gate.attn.kv_cache import PrefillCache

# Every captured shape's live outputs would otherwise stay allocated for the
# graph's lifetime (a shared pool reuses only memory that was freed before the
# next capture): at full context that is ~100 KB per token per shape, and 475
# prompts ran the b200 out of memory at the hundredth shape.  So the graphs
# write their K/V and their final hidden states into one set of static
# buffers sized min(SEQ_MAX, GRAPH_MAX_TOKENS), shared by every shape -- the
# latest prefill's cache is the only live one, which is the contract anyway.
# PHASE2_SEQ_MAX overrides the upper bound on capturable/storable length;
# replay_layers raises loudly if a capture-eligible prompt exceeds it.
SEQ_MAX = int(os.environ.get("PHASE2_SEQ_MAX", "65536"))
# Above this length replay_layers runs the plain compiled loop instead of
# capturing: every capture retains private-pool memory for the graph's
# lifetime, and long shapes are (a) compute-bound — launch overhead, the thing
# graphs remove, is noise there — and (b) usually UNIQUE (full-context arms:
# one length per prompt), so their graphs replay once and amortize nothing.
# At 500-prompt scale the full arms' single-use graphs alone pinned ~30 GB of
# pools across four resident gate models on an 80 GB A100.  10240 covers every
# exact phase2-length arm (T <= 8192 + padding) with room to spare.
GRAPH_MAX_TOKENS = int(os.environ.get("PHASE2_GRAPH_MAX_TOKENS", "10240"))
_POOL = None
_CAPTURE_STREAM = {}


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = torch.cuda.graph_pool_handle()
    return _POOL


def _capture_stream(device):
    """One reused side stream per device for every warmup + capture.  A fresh
    stream per capture left each graph's pool blocks associated with a stream
    that never ran again, so captures could not reuse the shared pool's freed
    blocks and the pool grew with every shape (~24 GB at the N=500 retime's
    hundreds of near-full-length shapes)."""
    idx = torch.device(device).index
    if idx not in _CAPTURE_STREAM:
        _CAPTURE_STREAM[idx] = torch.cuda.Stream(device=device)
    return _CAPTURE_STREAM[idx]


class _StaticStoreLayer(DynamicLayer):
    """A cache layer whose first write copies K/V into a static buffer (and
    exposes views of it), so a captured graph's cache lands at fixed addresses."""

    def __init__(self, config, buf_k, buf_v):
        super().__init__(config)
        self.buf_k, self.buf_v = buf_k, buf_v
        self.keys = buf_k[:, :, :0]
        self.values = buf_v[:, :, :0]
        self.is_initialized = True
        self.dtype, self.device = buf_k.dtype, buf_k.device

    def update(self, key_states, value_states, *args, **kwargs):
        if self.keys.shape[2] != 0:
            raise RuntimeError("_StaticStoreLayer takes one write per prefill")
        n = key_states.shape[2]
        self.buf_k[:, :, :n].copy_(key_states)
        self.buf_v[:, :, :n].copy_(value_states)
        self.keys = self.buf_k[:, :, :n]
        self.values = self.buf_v[:, :, :n]
        return self.keys, self.values


class _Entry:
    __slots__ = ("graph", "hs_in", "pos_in", "n_alloc")


def _bufs(model, first_layer, B, dtype, device):
    """The model's static graph buffers for this (batch, dtype): per-layer K/V
    + final hidden states (graph OUTPUTS, sized SEQ_MAX) and hidden states +
    positions (graph INPUTS, sized to the largest capturable shape).  All
    shared by every captured shape: a per-graph zeros_like input at hundreds
    of captured shapes was ~15 GB of dead weight at the N=500 retime (only
    the shape being replayed ever holds live data)."""
    cfg = model.config.get_text_config(decoder=True)
    bufs = model.__dict__.setdefault("_layer_graph_bufs", {})
    key = (B, dtype)
    if key not in bufs:
        n_kv = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        # Only CAPTURED shapes touch these buffers -- shapes above
        # GRAPH_MAX_TOKENS run the plain loop against the caller's own cache
        # (replay_layers returns before reaching them) -- so they are sized to
        # the largest capturable shape, not SEQ_MAX (4 GB/model of K/V at the
        # retime's SEQ_MAX=41088, of which everything past GRAPH_MAX_TOKENS
        # was unreachable).
        n_cap = min(SEQ_MAX, GRAPH_MAX_TOKENS)
        # static K/V only for FULL-ATTENTION layers: hybrid (Qwen3.5) DeltaNet
        # layers have no per-token KV cache — allocating/fixing-up buffers for
        # them crashed on their LinearAttentionLayer cache entries
        kv = {i: (torch.empty(B, n_kv, n_cap, hd, dtype=dtype, device=device),
                  torch.empty(B, n_kv, n_cap, hd, dtype=dtype, device=device))
              for i in range(first_layer, len(model.model.layers))
              if getattr(model.model.layers[i], "self_attn", None) is not None}
        hs_out = torch.empty(B, n_cap, cfg.hidden_size, dtype=dtype, device=device)
        hs_in = torch.zeros(B, n_cap, cfg.hidden_size, dtype=dtype, device=device)
        pos_in = torch.zeros(B, n_cap, dtype=torch.long, device=device)
        bufs[key] = (kv, hs_out, hs_in, pos_in)
    return bufs[key]


def _static_cache(model, first_layer, B, dtype, device):
    """A PrefillCache whose layers >= first_layer store into the model's static
    K/V buffers (allocated once per batch size)."""
    cfg = model.config.get_text_config(decoder=True)
    kv, hs_out, _, _ = _bufs(model, first_layer, B, dtype, device)
    cache = PrefillCache(model.config)
    for i in range(first_layer, len(model.model.layers)):
        if type(cache.layers[i]).__name__ not in ("DynamicLayer", "_StoreFirstLayer"):
            continue
        cache.layers[i] = _StaticStoreLayer(cfg, *kv[i])
    return cache, hs_out


def run_layers(model, first_layer: int, cache, hs: torch.Tensor, pos: torch.Tensor, n_alloc: int) -> torch.Tensor:
    """The plain loop: rotary embeddings from `pos`, then layers first_layer..end."""
    inner = model.model
    pe = inner.rotary_emb(hs, pos)
    cache_pos = torch.arange(n_alloc, device=hs.device)
    for i in range(first_layer, len(inner.layers)):
        hs = inner.layers[i](
            hs,
            attention_mask=None,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
            position_embeddings=pe,
        )
    return hs


def replay_layers(model, first_layer: int, cache, hs: torch.Tensor, pos: torch.Tensor, n_alloc: int) -> torch.Tensor:
    """run_layers's result through a graph captured once per (first_layer, B, n_alloc).

    Copies hs / pos into the graph's inputs, replays, and points cache.layers
    [first_layer:] at the static K/V buffers (full padded length; the caller
    cuts them back).  Returns the static output buffer's first n_alloc rows."""
    if hs.shape[1] != n_alloc or pos.shape[1] != n_alloc:
        raise ValueError(f"replay_layers: hs {tuple(hs.shape)} / pos {tuple(pos.shape)} are not padded to {n_alloc}")
    if n_alloc > GRAPH_MAX_TOKENS:
        # compute-bound and usually single-use: run the same compiled layers
        # directly (identical kernels and numerics, no pool retention)
        return run_layers(model, first_layer, cache, hs, pos, n_alloc)
    if n_alloc > SEQ_MAX:
        raise ValueError(f"replay_layers: {n_alloc} tokens exceed SEQ_MAX={SEQ_MAX}")
    B, dtype, device = hs.shape[0], hs.dtype, hs.device
    reg = model.__dict__.setdefault("_layer_graphs", {})
    key = (first_layer, B, n_alloc, dtype)
    e = reg.get(key)
    if e is None:
        e = _Entry()
        e.n_alloc = n_alloc
        _, _, hs_in_buf, pos_in_buf = _bufs(model, first_layer, B, dtype, device)
        e.hs_in = hs_in_buf[:, :n_alloc]
        e.pos_in = pos_in_buf[:, :n_alloc]
        e.hs_in.zero_()
        e.pos_in.zero_()
        # Warm up on a side stream (kernel loading, inductor's first calls at
        # this shape, with the same cache class the capture will use so dynamo
        # sees the same guards), then capture into the shared pool.
        s = _capture_stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(2):
                c, hs_out = _static_cache(model, first_layer, B, dtype, device)
                hs_out[:, :n_alloc].copy_(run_layers(model, first_layer, c, e.hs_in, e.pos_in, n_alloc))
        torch.cuda.current_stream(device).wait_stream(s)
        c, hs_out = _static_cache(model, first_layer, B, dtype, device)
        e.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(e.graph, pool=_pool(), stream=s):
            hs_out[:, :n_alloc].copy_(run_layers(model, first_layer, c, e.hs_in, e.pos_in, n_alloc))
        reg[key] = e
    e.hs_in.copy_(hs)
    e.pos_in.copy_(pos)
    e.graph.replay()
    kv, hs_out, _, _ = model._layer_graph_bufs[(B, dtype)]
    for i in range(first_layer, len(model.model.layers)):
        if i not in kv:
            continue
        dst = cache.layers[i]
        buf_k, buf_v = kv[i]
        if not dst.is_initialized:
            dst.lazy_initialization(buf_k, buf_v)
        dst.keys, dst.values = buf_k[:, :, :n_alloc], buf_v[:, :, :n_alloc]
    return hs_out[:, :n_alloc]


def graph_count(model) -> int:
    return len(model.__dict__.get("_layer_graphs", {}))


def release(model) -> int:
    """Free every captured graph and every static buffer, all batch sizes.

    The static K/V buffers are allocated per (batch size, dtype) and retained
    for the model's lifetime, and the graphs' pool blocks live as long as the
    graphs do.  A caller that RETRIES AT A SMALLER BATCH after an OOM
    (bench_grouped_eval's adaptive halving) must release the old size's state
    first: otherwise the retry allocates a fresh buffer set ON TOP of the
    retained one and every halving step makes the memory pressure strictly
    worse — full-context arms then die at the bs=1 floor (~56 GB of B=8
    buffers on Qwen3-1.7B, plus 4/2/1-sized sets, on an 80 GB A100).
    Re-capture after release is automatic, paid once per shape.

    Returns the number of graphs released."""
    global _POOL
    reg = model.__dict__.pop("_layer_graphs", {})
    n = len(reg)
    for e in reg.values():
        del e.graph          # a graph's pool blocks free when the graph does
    reg.clear()
    model.__dict__.pop("_layer_graph_bufs", None)
    _POOL = None             # fresh pool for any future capture
    _CAPTURE_STREAM.clear()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return n

#!/bin/bash
# Stage FlashAttention's CuTe-DSL kernels (flash_attn/cute from the flash-attn 2.8.3
# sdist: pure Python, JIT-compiled by nvidia-cutlass-dsl) outside the Python env, for
# the aug-fa4 Phase-2 backend (flex_attention_utils). Nothing is built; the CUDA
# extension the sdist would compile is not needed -- flash_attn/__init__.py is
# replaced by an empty one so the cute package imports on its own.
#
#   bash kernels/bench/fa4_setup.sh   # stages under ~/.cache/tokengate/fa4
#                                     # (FA4_ROOT / TOKENGATE_FA4_SITE override)
#
# Pins: nvidia-cutlass-dsl 4.1.0 (the release flash-attn 2.8.3's cute code targets;
# 4.7.x no longer ships cutlass.cute as Python), cuda-bindings 12.x.
set -euo pipefail
PY="${PY:-python}"
ROOT="${FA4_ROOT:-$HOME/.cache/tokengate/fa4}"
SITE=$ROOT/site41
mkdir -p "$ROOT" "$SITE"
cd "$ROOT"
[ -f flash_attn-2.8.3.post1.tar.gz ] || "$PY" -m pip download flash-attn==2.8.3.post1 --no-deps --no-binary :all: -d .
[ -f nvidia_cutlass_dsl-4.1.0-cp312-cp312-manylinux_2_28_x86_64.whl ] || "$PY" -m pip download nvidia-cutlass-dsl==4.1.0 --no-deps -d .
"$PY" -m pip install --no-deps --quiet --target "$SITE" nvidia_cutlass_dsl-4.1.0-*.whl
"$PY" -m pip install --quiet --target "$SITE" "cuda-bindings>=12.8,<13"
mkdir -p src && tar xzf flash_attn-2.8.3.post1.tar.gz -C src --wildcards "flash_attn-2.8.3.post1/flash_attn/cute/*"
mkdir -p "$SITE/flash_attn" && cp -r src/flash_attn-2.8.3.post1/flash_attn/cute "$SITE/flash_attn/" && : > "$SITE/flash_attn/__init__.py"
# Two lines of the sm100 forward: its head_dim-192 shared-memory scheme (K and V
# stages of unequal width sharing one buffer, the output tile overlapping Q) is
# what lets 192 fit in 227 KB; enabling the same scheme for 144 and 160 makes
# those launch too. head_dim 160 carries the gate's bias columns at +12% of the
# plain kernel at 22k tokens against +16% for 192.
"$PY" - "$SITE/flash_attn/cute/flash_fwd_sm100.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
for old, new in [
    ("self.overlap_sO_sQ = self.head_dim_padded == 192 and self.head_dim_v_padded >= 64",
     "self.overlap_sO_sQ = self.head_dim_padded in (144, 160, 192) and self.head_dim_v_padded >= 64"),
    ("self.uneven_kv_smem = self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3",
     "self.uneven_kv_smem = self.head_dim_padded in (144, 160, 192) and self.head_dim_v_padded == 128 and self.kv_stage == 3"),
]:
    if new in s:
        continue
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w").write(s)
print("flash_fwd_sm100.py: head_dim 144/160 enabled")
PYEOF
echo "staged: $SITE (import via site.addsitedir; the .pth files there matter)"

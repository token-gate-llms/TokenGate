import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
VERSION = "2.8.3.post1"
SDIST = pathlib.Path(os.environ.get("FA2_SDIST", f"~/.cache/tokengate/fa4/flash_attn-{VERSION}.tar.gz")).expanduser()
BUILD = pathlib.Path(os.environ.get("FA2BIAS_BUILD", HERE / "build")).expanduser()
SRC = BUILD / f"flash_attn-{VERSION}"
sys.path.insert(0, str(HERE))
from site_path import ABI, BUILD_INFO, SITE as _SITE  # noqa: E402
SITE = pathlib.Path(_SITE)
NAME = "flash_attn_keybias_cuda"

# Conda-style envs carry nvcc inside the env prefix; point CUDA_HOME there only
# when that's actually the case, otherwise leave discovery to
# torch.utils.cpp_extension (CUDA_HOME, PATH, /usr/local/cuda).
if "CUDA_HOME" not in os.environ and os.path.exists(os.path.join(sys.prefix, "bin", "nvcc")):
    os.environ["CUDA_HOME"] = sys.prefix
os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("FA2BIAS_ARCHS", "8.0+PTX")
os.environ.setdefault("MAX_JOBS", str(max(1, (os.cpu_count() or 2) - 1)))

BUILD.mkdir(parents=True, exist_ok=True)
# The unpacked tree is reused across builds only while it was produced by THIS patch set (the
# patch file, this script's source edits, the head-dim option). A tree left by an older
# build.py would otherwise compile into a .so that lacks the newer entry points while being
# stamped with the current ABI.
PATCHSET = hashlib.sha256((HERE / "keybias.patch").read_bytes() + pathlib.Path(__file__).read_bytes()
                          + os.environ.get("FA2BIAS_HDIM256", "1").encode()).hexdigest()
MARK = SRC / ".keybias_patchset"
if SRC.exists() and (not MARK.exists() or MARK.read_text() != PATCHSET):
    print(f"[build] {SRC} was unpacked by a different patch set: rebuilding from the sdist", flush=True)
    shutil.rmtree(SRC)
    shutil.rmtree(BUILD / "ext", ignore_errors=True)
if not SDIST.exists():
    SDIST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", f"flash-attn=={VERSION}", "--no-deps",
                    "--no-binary", ":all:", "-d", str(SDIST.parent)], check=True)
if not SRC.exists():
    with tarfile.open(SDIST) as t:
        t.extractall(BUILD, filter="data")
    subprocess.run(["patch", "-p1", "-s", "-i", str(HERE / "keybias.patch")], cwd=SRC, check=True)
    print(f"[build] unpacked {SDIST.name} and applied keybias.patch", flush=True)
    # The keybias.patch hardcodes head_dim=128 in run_mha_fwd (bf16-only). The
    # Qwen3.5 hybrid needs head_dim=256, so generalize the fwd dispatch to
    # {128, 256}. (bwd stays 128-only — inference never calls it.)
    # The variable-length entry points carry the bias too: one row of (num_heads, max_seqlen_k)
    # per packed sequence, indexed by the key's position inside its own sequence (the kernel
    # reads bidb and the block's column offset, both already per sequence under cu_seqlens).
    # Training packs the short and the full Phase-2 pass into one launch with it.
    api = SRC / "csrc" / "flash_attn" / "flash_api.cpp"
    txt = api.read_text()
    swaps = [
        ('    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);\n\n    if (max_seqlen_k > 0) {\n',
         '    set_params_alibi(params, alibi_slopes_, batch_size, num_heads, max_seqlen_k);\n\n    if (max_seqlen_k > 0) {\n'),
        ('    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);\n\n    if (max_seqlen_q > 0) {\n        launch(params, stream);\n',
         '    at::Tensor dbias;\n'
         '    if (alibi_slopes_.has_value()) { dbias = torch::zeros({batch_size, num_heads, max_seqlen_k}, opts.dtype(at::kFloat)); }\n'
         '    set_params_alibi(params, alibi_slopes_, batch_size, num_heads, max_seqlen_k);\n'
         '    params.dbias_ptr = alibi_slopes_.has_value() ? dbias.data_ptr() : nullptr;\n'
         '    params.dbias_batch_stride = alibi_slopes_.has_value() ? dbias.stride(0) : 0;\n'
         '    params.dbias_head_stride = alibi_slopes_.has_value() ? dbias.stride(1) : 0;\n\n'
         '    if (max_seqlen_q > 0) {\n        launch(params, stream);\n'),
        ('    return { dq, dk, dv, softmax_d };\n', '    return { dq, dk, dv, softmax_d, dbias };\n'),
    ]
    for old, new in swaps:
        if txt.count(old) != 1:
            raise SystemExit(f"[build] varlen site not found exactly once: {old[:60]!r}")
        txt = txt.replace(old, new)
    api.write_text(txt)
    print("[build] key bias enabled on varlen_fwd / varlen_bwd", flush=True)
    if os.environ.get("FA2BIAS_HDIM256", "1") == "1":
        api = SRC / "csrc" / "flash_attn" / "flash_api.cpp"
        txt = api.read_text()
        old = ('    TORCH_CHECK(params.is_bf16 && params.d == 128, "this build supports bf16 with head_dim 128 only");\n'
               '    using elem_type = cutlass::bfloat16_t;\n'
               '    constexpr int kHeadDim = 128;\n'
               '    BOOL_SWITCH(params.is_causal, Is_causal, [&] {\n'
               '        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don\'t set it num_splits == 0\n'
               '            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);\n'
               '        } else {\n'
               '            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);\n'
               '        }\n'
               '    });')
        new = ('    TORCH_CHECK(params.is_bf16 && (params.d == 128 || params.d == 256),\n'
               '                "this build supports bf16 with head_dim 128 or 256 only");\n'
               '    using elem_type = cutlass::bfloat16_t;\n'
               '    BOOL_SWITCH(params.is_causal, Is_causal, [&] {\n'
               '        if (params.d == 128) {\n'
               '            if (params.num_splits <= 1 && !force_split_kernel)\n'
               '                run_mha_fwd_<elem_type, 128, Is_causal>(params, stream);\n'
               '            else run_mha_fwd_splitkv_dispatch<elem_type, 128, Is_causal>(params, stream);\n'
               '        } else {\n'
               '            if (params.num_splits <= 1 && !force_split_kernel)\n'
               '                run_mha_fwd_<elem_type, 256, Is_causal>(params, stream);\n'
               '            else run_mha_fwd_splitkv_dispatch<elem_type, 256, Is_causal>(params, stream);\n'
               '        }\n'
               '    });')
        if old not in txt:
            raise SystemExit("[build] flash_api.cpp fwd-dispatch block not found — the "
                             "keybias.patch layout changed; update the hdim256 generalization.")
        api.write_text(txt.replace(old, new))
        print("[build] generalized run_mha_fwd dispatch to head_dim {128,256}", flush=True)

MARK.write_text(PATCHSET)

from torch.utils.cpp_extension import load  # noqa: E402

csrc = SRC / "csrc" / "flash_attn"
_fwd = [
    "flash_fwd_hdim128_bf16_sm80.cu", "flash_fwd_hdim128_bf16_causal_sm80.cu",
    "flash_fwd_split_hdim128_bf16_sm80.cu", "flash_fwd_split_hdim128_bf16_causal_sm80.cu",
    "flash_bwd_hdim128_bf16_sm80.cu", "flash_bwd_hdim128_bf16_causal_sm80.cu",
]
if os.environ.get("FA2BIAS_HDIM256", "1") == "1":
    # fwd + split-fwd only (inference); bwd stays 128-only (never called at eval).
    _fwd += [
        "flash_fwd_hdim256_bf16_sm80.cu", "flash_fwd_hdim256_bf16_causal_sm80.cu",
        "flash_fwd_split_hdim256_bf16_sm80.cu", "flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    ]
sources = [csrc / "flash_api.cpp"] + [csrc / "src" / f for f in _fwd]
ext_dir = BUILD / "ext"
ext_dir.mkdir(exist_ok=True)
load(
    name=NAME,
    sources=[str(s) for s in sources],
    extra_include_paths=[str(csrc), str(csrc / "src"), str(SRC / "csrc" / "cutlass" / "include"),
                         str(pathlib.Path(sys.prefix) / "targets" / "x86_64-linux" / "include")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DFLASHATTENTION_DISABLE_DROPOUT", "-DFLASHATTENTION_DISABLE_SOFTCAP",
        "-DFLASHATTENTION_DISABLE_UNEVEN_K", "-DFLASHATTENTION_DISABLE_LOCAL",
        "--threads", "2",
    ],
    build_directory=str(ext_dir),
    verbose=True,
    is_python_module=False,
)
so = next(ext_dir.glob(f"{NAME}*.so"))
SITE.mkdir(parents=True, exist_ok=True)
shutil.copy2(so, SITE / so.name)
(SITE / BUILD_INFO).write_text(json.dumps({"abi": ABI, "flash_attn": VERSION, "archs": os.environ["TORCH_CUDA_ARCH_LIST"]}))
print(f"[build] {so.name} -> {SITE}", flush=True)

import os

# The one place that says where the built extension lives: build.py writes it there and
# keybias.py loads it from there. FA2BIAS_SITE overrides both at once.
SITE = os.path.expanduser(os.environ.get("FA2BIAS_SITE", "~/.cache/tokengate/fa2bias/site"))
BUILD_INFO = "BUILD_INFO.json"
# Bumped when the extension gains an entry point callers rely on. 2: key bias on varlen_fwd / varlen_bwd.
ABI = 2

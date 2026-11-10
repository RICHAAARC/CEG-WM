"""Build the fixed import-only compatibility continuation, with no user switch."""
from scripts.build_stage_protection_v2_notebook import build

if __name__ == "__main__":
    build(compatibility=True)

"""Build the fixed continuation delivery, with no user-editable run switch."""
from scripts.build_stage_protection_v2_notebook import build

if __name__ == "__main__":
    build(continuation=True)

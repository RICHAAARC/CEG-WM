"""Fixed entry for the observed load-only private-symbol ImportError."""
from experiments.run_stage_protection_v2_resume import main

if __name__ == "__main__":
    raise SystemExit(main(compatibility=True))

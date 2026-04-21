# Released Results

Place the benchmark outputs that you want to publish together with the code in this folder.

Recommended files for release:

- `paper_table_main_multiseed.csv`
- `paper_table_ablation.csv`
- `paper_table_main_multiseed_extended.csv`
- `paper_table_deployability_extended.csv`
- selected `summary_results.csv` files for each run
- optional `paper_subset_inventory.csv` files for transparency

To avoid leaking credentials or machine-specific information, do not upload API keys, local runtime logs containing secrets, or private filesystem paths.

A practical workflow is to keep raw large intermediate outputs under `runs/` locally, then copy only the paper-facing CSV files and inventories into `released_results/` before pushing the repository to GitHub.

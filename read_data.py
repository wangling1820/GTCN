import itertools
from read_data_utils import *

src_files = [
    r'./data/alpha/alpha.txt',
]


train_rates = [0.60]        # Proportion of training data
val_rates = [0.10]          # Proportion of validation data
test_rates = [0.30]         # Proportion of testing data

edge_life_windows = [20]    # Time window for edge life span
make_symmetric_flags = [True]  # Whether to symmetrize adjacency matrices
edge_life_flags = [False]      # Whether to apply edge life modeling

print("Source files:", src_files)

param_grid = itertools.product(
    src_files,
    train_rates,
    val_rates,
    test_rates,
    edge_life_flags,
    edge_life_windows,
    make_symmetric_flags
)

for setting in param_grid:
    src_file, train_rate, val_rate, test_rate, edge_life, edge_life_window, make_symmetric = setting

    print(f"\n Current configuration:\n"
          f" - Dataset file: {src_file}\n"
          f" - Split ratio (train/val/test): {train_rate:.2f}/{val_rate:.2f}/{test_rate:.2f}\n"
          f" - Edge life: {'Enabled' if edge_life else 'Disabled'} (window={edge_life_window})\n"
          f" - Make symmetric: {make_symmetric}\n")

    try:
        read_data(
            src_file=src_file,
            train_rate=train_rate,
            val_rate=val_rate,
            test_rate=test_rate,
            edge_life=edge_life,
            edge_life_window=edge_life_window,
            make_symmetric=make_symmetric
        )
        print("Data successfully loaded.")
    except Exception as e:
        print(f"Data loading failed: {e}")

print("\n All tasks completed successfully!")

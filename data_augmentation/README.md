## Data Augmentation

The CSV files are only representatives of the entire augmented dataset.

The entire dataset takes up over 80MB so needs to be generated locally using corresponding scripts.

**Explanation:**

| File | Description |
| - | - |
| `data_augmentation.py` | Python script to generate the augmented data in `aug_set_metadata.csv` and `aug_set.csv` |
| `aug_set_metadata.csv` | Holds the result of the augmentation script (if not ran, stores example output) on the astronomical objects' metadata |
| `aug_set_metadata.csv` | Holds the result of the augmentation script (if not ran, stores example output) on the timeseries data |
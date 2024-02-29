# CNNTDANet



## Directory structure



## Key Features

### Efficient metric logging system
- 


## Dataset Setup Guidline

For optimal use with this repository, please structure your datasets as follows:


```bash
data/
    ├── dataset_name1/
    │   ├── metadata
    │   │   └── dataset1_metadata.csv
    │   ├── images/
    │   │   ├── train/
    │   │   ├── test/
    │   │   └── ground_truth/ 
    │   └── topology/
    │       ├── train/
    │       │   ├──betti_curve/
    │       │   └──euler_curve/
    │       └── test/
    │           ├──betti_curve/
    │           └──euler_curve/
    │        
    └── dataset_name2/
        ├── metadata/
        │   └── dataset2_metadata.csv
        ├── images/
        │   └── ...
        └── topology/
            └── ...
```

**Important Points:**

- `metadata`: Contains `.csv` files with image IDs, labels, etc.
- `images`: Divided into train, test, and ground_truth (if your task is ).
- `topology`: Holds topological vectors, separated into train and test.

Ensure this structure to enable the program to accurately process your data.
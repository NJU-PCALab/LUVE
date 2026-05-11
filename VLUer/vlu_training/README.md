# VLU training code
Use `dist.sh` to execute the training code. Relevant configuration files are located in the `configs` directory.

For reference code regarding training data generation, please see `datasets/scripts/`. The directory structure for the training data is as follows:

```text
dataset/
├── video/
│   └── 0001.mp4
├── HR_latent/
│   └── 0001.npy
└── LR_latent/
    └── X1.5/
        └── 0001.npy
```
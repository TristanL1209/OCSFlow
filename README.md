# OCSFlow

Official implementation of **“Reformulating Hyperspectral Image Classification as Observation-Conditioned Class-State Flow on Soft Supertoken Graphs.”**

This release contains the complete OCSFlow training and evaluation path used for the four benchmarks in the paper. All datasets use full-image soft-supertoken construction; ground-truth labels are used only through the training split selected by `train_mask`.

## Environment

The code has been tested with:

- Python 3.9.21
- PyTorch 2.5.1
- CUDA 12.4
- NumPy 2.0.2
- SciPy 1.13.1
- h5py 3.14.0
- PyYAML 6.0.2

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA-enabled installation, install the PyTorch build appropriate for your CUDA runtime before installing the remaining requirements if necessary.

## Datasets

Download the original MATLAB files from their respective providers:

- Indian Pines, Pavia University, and Kennedy Space Center: [UPV/EHU Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
- WHU-Hi HanChuan: [WHU RSIDEA resource page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm)

Place the files under `datasets/` with the following names:

```text
datasets/
├── Indian_pines_corrected.mat
├── Indian_pines_gt.mat
├── PaviaU.mat
├── PaviaU_gt.mat
├── KSC.mat
├── KSC_gt.mat
├── WHU_Hi_HanChuan.mat
└── WHU_Hi_HanChuan_gt.mat
```

The default path is `./datasets`. To use another location, update `common.data_root` in `configs/ocsflow.yaml`.

Dataset files are not included in this repository. Please follow the terms of the original data providers.

## Training and evaluation

Each command trains one model using 30 labeled samples per class for training, 10 per class for validation, and the remaining labeled samples for testing. The default random seed is 42.

```bash
python train.py --dataset IP  --seed 42 --config configs/ocsflow.yaml
python train.py --dataset PU  --seed 42 --config configs/ocsflow.yaml
python train.py --dataset KSC --seed 42 --config configs/ocsflow.yaml
python train.py --dataset HC  --seed 42 --config configs/ocsflow.yaml
```

The program prints the final evaluation metrics as JSON. CUDA is selected automatically when available; otherwise the code runs on CPU. Set `common.device` in the configuration file to override this behavior.

## Reproducibility notes

- Randomness in Python, NumPy, and PyTorch is seeded from `--seed`.
- Dataset normalization statistics are computed from training pixels only.
- Soft supertokens are constructed over the full image for every dataset.
- Evaluation noise uses a fixed seed offset configured in `configs/ocsflow.yaml`.

## Repository structure

```text
configs/        Model and training configuration
data/           Dataset loading, splitting, and preprocessing
evaluation/     Metrics and evaluation routines
models/         OCSFlow model components
training/       Training and flow-matching routines
train.py        Command-line entry point
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

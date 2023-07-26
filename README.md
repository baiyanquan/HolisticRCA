# HolisticRCA

Artifacts accompanying HolisticRCA, a framework for failure root cause analysis in cloud-native systems from a holistic perspective. 

## Requirements

### Dependencies

````
cd ./code
pip install -r requirements.txt
````

### Our Test Sandbox

- Intel(R) Xeon(R) Gold 6226R CPU
- 128GB RAM
- NVIDIA GeForce RTX 4090 GPU
- Ubuntu 20.04.6 LTS
- Docker version 19.03.12
- Python 3.10 (Docker image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel)

## Folder Structure

Note: we have split the zip file for uploading. The following commands need to be input for the working folder:
````
zip model_split.zip -s=0 --out model.zip
unzip model.zip
````
Then the temp data also needs to be downloaded from https://anonymous.4open.science/r/HolisticRCATempData-D9E4.

Finally the ``model`` and ``temp_data`` folders need be placed following the structure of the working folder:
````
.
├── README.md
├── code                                          
│   ├── data_filter                             preprocess data
│   │   ├── CCF_AIOps_challenge_2022            preprocess dataset A
│   │   ├── ICASSP_AIOps_challenge_2022         preprocess dataset B
│   │   └── Eadro_TT_and_SN                     preprocess dataset C
│   ├── HolisticRCA                             the main model of the work
│   │   ├── ablation                            models for ablation study
│   │   ├── base                                base classes for model construction
│   │   ├── config                              configuration of file paths
│   │   ├── data_loader                         load dataset
│   │   ├── dataset                             base class for dataset reader
│   │   ├── explain                             mask learning component (for resource entity localization and fault-related observability data localization)
│   │   ├── model                               main components except mask learning
│   │   ├── trainer                             perform model training
│   │   └── util                                RERG class and data transformation
│   ├── shared_util                             some basic util functions
│   ├── experiments_a.sh                        quick experiments for dataset A
│   ├── experiments_b.sh                        quick experiments for dataset B
│   ├── experiments_c.sh                        quick experiments for dataset C
│   └── requirements.txt
├── model                                       saved model data for reproduction
└── temp_data                                   saved temp data for reproduction
````

## Quick Start / Reproducibility

### Prerequisites

1. Prepare the Python packages in ``requirements.txt``.
2. Unzip ``model.zip`` and ``temp_data.zip``.

### Simple Result Checking

The saved model files are placed in ``model.zip``. Following the files ``experiments_a.sh``, ``experiments_b.sh``, or ``experiments_c.sh`` and comment out ``rca_data_trainer.train()`` in the corresponding training files. It will output the evaluation results (note that some file paths need to be changed).

### Running

````
cd ./code
bash experiments_a.sh
bash experiments_b.sh
bash experiments_c.sh
````

## Raw Data

Since the raw data is too big, we list their links here, help for downloading:

- Dataset A: https://competition.aiops-challenge.com/home/competition/1496398526429724760. (Sometimes the page may be crashed, please visit https://www.bizseer.com/index.php?m=content&c=index&a=show&catid=25&id=83 for simple introduction).
- Dataset B: https://www.aiops.sribd.cn/home/introduction.
- Dataset C: https://doi.org/10.5281/zenodo.7615393.

## Statement

For the sake of preserving anonymity, we have removed all textual comments from the code, which will be added to the source code once the work is formally published.

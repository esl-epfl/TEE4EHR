

# [TimEHR: Image-based Time Series Generation for Electronic Health Records](https://arxiv.org/abs/2402.06318)

[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE) 
<img src="https://img.shields.io/badge/python-3.9.7-blue">
<img src="https://img.shields.io/badge/pytorch-2.2.2-orange">

> **Abstract:** Time series in Electronic Health Records (EHRs) present unique challenges for generative models, such as irregular sampling, missing values, and high dimensionality. In this paper, we propose a novel generative adversarial network (GAN) model, **TimEHR**, to generate time series data from EHRs. In particular, TimEHR treats time series as images and is based on two conditional GANs. The first GAN generates missingness patterns, and the second GAN generates time series values based on the missingness pattern. Experimental results on three real-world EHR datasets show that TimEHR outperforms state-of-the-art methods in terms of fidelity, utility, and privacy metrics.

## Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
<!-- - [Citation](#citation) -->

## Installation
Clone the repository, create a virtual environment (`venv` or `conda`), and install the required packages using `pip`:
```bash
# clone the repository
git clone https://github.com/hojjatkarami/TimEHR.git
cd TimEHR

# using virtualenv
python3 -m venv test2
source "/mlodata1/hokarami/Machine-Learning-Collection/ML/Pytorch/GANs/4. WGAN-GP/test2/bin/activate"

# using conda
conda create --name TimEHR python=3.9.7 --yes
conda activate TimEHR

# install the required packages
pip install -r requirements.txt
```


## Datasets
We used three real-world EHRs datasets as well as simulated data in our experiments:


| Dataset Name | Size | Number of Features |
|--------------|------|--------------------|
| [PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) | 12k | 35 |
| [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) | 38k | 32 |
| [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) | 51k | 37 |
| Simulated Data | 10k | 16,32,64,128 |

We need to convert **irregularly-sampled time series** to images. Please refer to the [data](data) folder for more details on the datasets.

<p align="center">
  <img src="figures/git-data.png" alt="" height=200>
  <br>
  Converting time series to images.
</p>

## Quick Start

We use `hydra-core` library for managing all configuration parameters. You can change them from `configs/config.yaml`. 

We highly recommend using `wandb` for logging and tracking the experiments. Get your API key from [wandb](https://wandb.ai/authorize). Create a `.env` file in the root directory and add the following line:

```bash
WANDB_API_KEY=your_api_key
```
### Training


<p align="center">
  <img src="figures/git-train.png" alt=""  height=200>
  <br>
  Training Procedure.
</p>

The following command will train the model and generate synthetic time series for `P12-split0` (You should have prepared the data in the `data` folder before running):
```bash
python train.py
```
This will train TimEHR modules (CWGAN-GP and Pix2Pix) for the default configuration (P12 dataset, split0) and prints the generated dataframe. Modules are saved locally in `Results/{dataset}-s{split}/[CWGAN|Pix2Pix]/` folder as well as on wandb servers (`account_name/[CWGAN|PIXGAN]`).

### Evaluation
<p align="center">
  <img src="figures/git-eval.png" alt="" height=200>
  <br>
  Evaluation Procedure.
</p>

```bash
python eval.py Results/p12-s0
```

This will generate and evaluate synthetic time series for the trained models in the `Results/p12-s0` folder and save the results in a wandb project `TimEHR-Eval` as well as locally in the `Results/p12-s0/TimEHR-Eval` folder.

For a more in-depth tutorial on how to train, generate, evaluate, and visualize the synthetic data, please checkout our notebook [Tutorial.ipynb](Tutorial.ipynb).

# Replication of the results in the paper

To replicate the results in the paper, please follow the steps below:

1. Run the following commands: 
    ```bash
    python train.py -m data=p12 split=0,1,2,3,4
    python train.py -m data=mimic split=0,1,2,3,4
    python train.py -m data=p19 split=0,1,2,3,4 pix2pix.lambda_l1=100

    ```
2. Use `python eval.py Results/{dataset}-s{split}` for the evaluation. The results will be saved in wanbd dashboard (`account_name/TimEHR-Eval`).



## Citation
If you find this repo useful, please cite our paper via
```bibtex
@article{karami2024timehr,
  title={TimEHR: Image-based Time Series Generation for Electronic Health Records},
  author={Karami, Hojjat and Hartley, Mary-Anne and Atienza, David and Ionescu, Anisoara},
  journal={arXiv preprint arXiv:2402.06318},
  year={2024}
}
```




```bash
curl https://pyenv.run | bash


export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

pyenv install 3.9.7
pyenv global 3.9.7
python --version


```

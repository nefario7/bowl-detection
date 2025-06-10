<div align="center">

# Bowl Detection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A PyTorch Lightning-based object detection system for bowl detection.

## Assignment Report

[Chef Robotics.pdf](./Chef%20Robotics.pdf)

## Installation

```bash
# clone project
git clone https://github.com/nefariov7/bowl-detection
cd bowl-detection

# install and set Python version using pyenv
pyenv install 3.10.0
pyenv local 3.10.0

# create virtual environment
pyenv virtualenv 3.10 bowl-detection-env
pyenv activate bowl-detection-env

# install requirements
pip install -r requirements.txt
```

## Weights Download

To use pre-trained weights, download them from Google Drive and place them in the weights directory:

1. Download the weights from: https://drive.google.com/drive/folders/1mI8pIV6okx-tSr7I6SkaWMgeQvmw2u3w?usp=sharing

2. Extract and save the downloaded files to the `weights/` directory in your project:
   ```
   bowl-detection/
   └── weights/
       └── [downloaded weight files]
   ```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu model=retinanet

# train on GPU
python src/train.py trainer=gpu model=retinanet
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

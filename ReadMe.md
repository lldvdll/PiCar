# Data

download kaggle data and extract to data/ (note: never commit this directory!)
https://www.kaggle.com/competitions/machine-learning-in-science-ii-2026/data

# Setup

install miniconda if conda missing - https://www.anaconda.com/docs/getting-started/miniconda/main
windows (bash) add command: ```echo 'eval "$(/c/Users/$USERNAME/miniconda3/Scripts/conda.exe shell.bash hook)"' >> ~/.bashrc```

# Create Environment

```
conda create -n pi_car python=3.9
conda activate pi_car
pip install -r requirements.txt
```

# Weights and Biases


# Data preparation
prepare_dataset.py - runs label bias analysis and creates a new train file with weightings for even sampling over speed/angle joint distribution


# MLiS server connect
ssh [uni username]@mlis1@nottingham.ac.uk
ssh [uni username]@mlis2@nottingham.ac.uk

# Connecting from home
https://windows.cloud.microsoft/#/devices - VM service for home access

# Remote instructions
git --version  - install with apt if missing

...add ssh key stuff...

```
git clone git@github.com:lldvdll/PiCar.git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n pi_car python=3.9
conda activate pi_car
pip install -r PiCar/requirements.txt
```

- check gpu status ```nvidia-smi```
- pull latest code ```git pull```
- activate environment ```conda activate pi_car```
- install requirements ```pip install -r PiCar/requirements.txt```
- 
- 

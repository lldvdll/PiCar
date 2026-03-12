# Data

download kaggle data and extract to data/ (note: never commit this directory!)
https://www.kaggle.com/competitions/machine-learning-in-science-ii-2026/data

# Setup

install miniconda if conda missing - https://www.anaconda.com/docs/getting-started/miniconda/main
windows (bash) add command: echo 'eval "$(/c/Users/$USERNAME/miniconda3/Scripts/conda.exe shell.bash hook)"' >> ~/.bashrc

# Create Environment

conda create -n pi_car python=3.9
conda activate pi_car
pip install -r requirements.txt

# Weights and Biases


# Data preparation
prepare_dataset.py - runs label bias analysis and creates a new train file with weightings for even sampling over speed/angle joint distribution


# MLiS server connect
ssh [uni username]@mlis1@nottingham.ac.uk
ssh [uni username]@mlis2@nottingham.ac.uk

# Connecting from home
https://windows.cloud.microsoft/#/devices - VM service for home access
#!/bin/bash

# Activating Conda environment in WSL
echo "Activating Conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TORCHNILMENV
echo "Environment TORCHNILMENV activated"

# Check if Conda environment was activated
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to activate Conda environment!"
    exit 1
fi

python src/model/train_model.py

echo "Python script executed successfully."

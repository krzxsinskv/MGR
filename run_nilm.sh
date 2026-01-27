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

python src/model/evaluate_model.py --ds refit_h9 --app fridge --mdl models/2025-12-16_15-50_best_model.pth


echo "Python script executed successfully."

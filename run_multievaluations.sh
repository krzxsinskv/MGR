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

# Executing commands - case 5

python src/model/evaluate_model.py --ds ukdale_h1 --app fridge --mdl models/fridge_80_case5.pth
python src/model/other_maps.py --mdl models/fridge_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app fridge --mdl models/fridge_40_case5.pth
python src/model/other_maps.py --mdl models/fridge_40_case5.pth

python src/model/evaluate_model.py --ds ukdale_h1 --app microwave --mdl models/microwave_80_case5.pth
python src/model/other_maps.py --mdl models/microwave_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app microwave --mdl models/microwave_40_case5.pth
python src/model/other_maps.py --mdl models/microwave_40_case5.pth

python src/model/evaluate_model.py --ds ukdale_h1 --app television --mdl models/television_80_case5.pth
python src/model/other_maps.py --mdl models/television_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app television --mdl models/television_40_case5.pth
python src/model/other_maps.py --mdl models/television_40_case5.pth

python src/model/evaluate_model.py --ds ukdale_h1 --app kettle --mdl models/kettle_80_case5.pth
python src/model/other_maps.py --mdl models/kettle_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app kettle --mdl models/kettle_40_case5.pth
python src/model/other_maps.py --mdl models/kettle_40_case5.pth

python src/model/evaluate_model.py --ds ukdale_h1 --app dishwasher --mdl models/dishwasher_80_case5.pth
python src/model/other_maps.py --mdl models/dishwasher_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app dishwasher --mdl models/dishwasher_40_case5.pth
python src/model/other_maps.py --mdl models/dishwasher_40_case5.pth

python src/model/evaluate_model.py --ds ukdale_h1 --app washing_machine --mdl models/washing_machine_80_case5.pth
python src/model/other_maps.py --mdl models/washing_machine_80_case5.pth
python src/model/evaluate_model.py --ds ukdale_h1 --app washing_machine --mdl models/washing_machine_40_case5.pth
python src/model/other_maps.py --mdl models/washing_machine_40_case5.pth

echo "Python script executed successfully."

source ../venv/3.7/bin/activate
source env.sh
# python3 oatomobile/baselines/torch/cil/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model/cil --num_epochs=20
nohup python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model/dim --num_epochs=200
# python3 oatomobile/baselines/torch/cil/train.py --dataset_dir=data-oatml/examples --output_dir=data-oatml/model/cil --num_epochs=3
# python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/examples --output_dir=data-oatml/model/dim --num_epochs=3

# python3 ray/ray-train.py

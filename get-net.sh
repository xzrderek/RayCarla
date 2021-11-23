source ../venv/3.7/bin/activate
source env.sh
python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/examples --output_dir=data-oatml/model-net --log_net=True --num_epochs=1

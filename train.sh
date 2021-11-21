source ../venv/3.7/bin/activate
source env.sh

STARTTIME=$(date +%s)
# python3 oatomobile/baselines/torch/cil/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model/cil --num_epochs=20
python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model/dim --num_epochs=200
ENDTIME=$(date +%s)
echo "It takes $($ENDTIME - $STARTTIME) seconds to complete this task..."
# Non-Ray Way

# Non-Ray Way, small dataset, for debugging
# python3 oatomobile/baselines/torch/cil/train.py --dataset_dir=data-oatml/examples --output_dir=data-oatml/model/cil --num_epochs=3
# python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/examples --output_dir=data-oatml/model/dim --num_epochs=3


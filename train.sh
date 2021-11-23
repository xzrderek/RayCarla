source ../venv/3.7/bin/activate
source env.sh

STARTTIME=$(date +%s)
# python3 oatomobile/baselines/torch/cil/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model-slow-cil --num_epochs=200
python3 oatomobile/baselines/torch/dim/train.py --dataset_dir=data-oatml/processed --output_dir=data-oatml/model-slow-200 --num_epochs=200
ENDTIME=$(date +%s)
echo "Training time: $(($ENDTIME - $STARTTIME)) seconds"

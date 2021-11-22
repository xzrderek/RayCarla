source ../venv/3.7/bin/activate
source env.sh

# Ray Way
STARTTIME=$(date +%s)
# Use examples for debugging
# python3 ray/ray-train.py --num_cpus=8 --num_gpus=8 --num_workers=8 --dataset_dir=data-oatml/examples --output_dir=data/model --epochs=10
# Use processed for real training
python3 ray/ray-train.py --num_cpus=8 --num_gpus=8 --num_workers=8 --dataset_dir=data-oatml/processed --output_dir=data-oatml/model --epochs=200
ENDTIME=$(date +%s)
echo "Training time: $(($ENDTIME - $STARTTIME)) seconds"
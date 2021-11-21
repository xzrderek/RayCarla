source ../venv/3.7/bin/activate
source env.sh

# Ray Way
STARTTIME=$(date +%s)
python3 ray/ray-train.py --num_cpus=8 --num_gpus=8 --num_workers=8 --dataset_dir=data-oatml/processed --epochs=200
ENDTIME=$(date +%s)
echo "It takes $($ENDTIME - $STARTTIME) seconds to complete this task..."

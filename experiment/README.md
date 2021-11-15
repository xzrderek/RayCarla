# Quick start
Please follow experiment/experiments.ipynb,

------------
# Old Readme
## virtualenv
```
source venv/bin/activate
source .env
```

## Prepare dataset
### Train Dataset
```
python oatomobile/baselines/rulebased/autopilot/run.py -output_dir=data/raw/train --town=Town04 --max_episode_steps=1000

python experiment/postprocess.py --input_dir=data/raw/train --output_dir=data/dataset/train
```
### Validation Dataset
```
python oatomobile/baselines/rulebased/autopilot/run.py -output_dir=data/raw/val --town=Town03 --max_episode_steps=500

python experiment/postprocess.py --input_dir=data/raw/val --output_dir=data/dataset/val
```

## Train
### Dim
```
python oatomobile/baselines/torch/dim/train.py -dataset_dir=data/dataset --output_dir=data/model/dim --num_epochs=100
```
### Cil
```
python oatomobile/baselines/torch/cil/train.py -dataset_dir=data/dataset --output_dir=data/model/cil --num_epochs=100
```
## RIP
TODO

## Drive
### Dim
python dim.py
## Cil
#python cil.py
### RIP
TODO

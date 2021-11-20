from dotenv import load_dotenv
load_dotenv()

import torch
from oatomobile.baselines.torch import ImitativeModel
from oatomobile.baselines.torch import BehaviouralModel
from oatomobile.baselines.torch import DIMAgent
from oatomobile.baselines.torch import RIPAgent
from oatomobile.baselines.torch import CILAgent

ckpts = [ \
  "data-oatml/model-200/dim/ckpts/model-196.pt",
  "data-oatml/model-200/dim/ckpts/model-192.pt",
  "data-oatml/model-200/dim/ckpts/model-188.pt",
  "data-oatml/model-200/dim/ckpts/model-184.pt", 
  ] # Paths to the model checkpoints.
models = [ImitativeModel() for _ in range(4)]
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
  
log_dir = "data-oatml/benchmarks/rip"
subtasks_id = None #"Hills0"

from oatomobile.benchmarks import carnovel

# drive!
on_ray = False
if on_ray:
  import ray
  ray.init(num_cpus=16, num_gpus=8, local_mode=True)
  from oatomobile.benchmarks import evaluate_on_ray   
  results = evaluate_on_ray(agent_fn=RIPAgent, log_dir=log_dir, render=False, monitor=True, subtasks_id=subtasks_id, models=models, algorithm="WCM")
  print(ray.get(results))
else:
  carnovel.evaluate(agent_fn=RIPAgent, log_dir=log_dir, render=False, monitor=True, subtasks_id=subtasks_id, models=models, algorithm="WCM")

# plot routes 
carnovel.plot_benchmark(log_dir)

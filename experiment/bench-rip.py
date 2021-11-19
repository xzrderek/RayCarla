from dotenv import load_dotenv
load_dotenv()

import torch
from oatomobile.baselines.torch import ImitativeModel
from oatomobile.baselines.torch import BehaviouralModel
from oatomobile.baselines.torch import DIMAgent
from oatomobile.baselines.torch import RIPAgent
from oatomobile.baselines.torch import CILAgent
from oatomobile.benchmarks import carnovel

ckpts = [ \
  "data-oatml/model/dim/ckpts/model-16.pt",
  "data-oatml/model/dim/ckpts/model-12.pt",
  "data-oatml/model/dim/ckpts/model-8.pt",
  "data-oatml/model/dim/ckpts/model-4.pt", 
  ] # Paths to the model checkpoints.
models = [ImitativeModel() for _ in range(4)]
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
log_dir = "data-oatml/benchmarks/rip"

carnovel.evaluate(agent_fn=RIPAgent, log_dir=log_dir, render=False, monitor=True, subtasks_id=None, models=models, algorithm="WCM")
carnovel.plot_benchmark(log_dir)

from dotenv import load_dotenv
load_dotenv()

import torch
from oatomobile.baselines.torch import ImitativeModel
from oatomobile.baselines.torch import BehaviouralModel
from oatomobile.baselines.torch import DIMAgent
from oatomobile.baselines.torch import RIPAgent
from oatomobile.baselines.torch import CILAgent
from oatomobile.benchmarks import carnovel

model = ImitativeModel() #ImitativeModel()
model.load_state_dict(torch.load("data-oatml/model/dim/ckpts/model-16.pt"))
log_dir = "data-oatml/benchmarks/dim"

carnovel.evaluate(agent_fn=DIMAgent, log_dir=log_dir, render=False, monitor=True, subtasks_id=None, model=model)
carnovel.plot_benchmark(log_dir)

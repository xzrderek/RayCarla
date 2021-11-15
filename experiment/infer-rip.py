# Imitation-learners.
import torch
import oatomobile.baselines.torch
import oatomobile
from oatomobile.envs import CARLAEnv

ckpts = ["data/model/dim/ckpts/model-0.pt"] * 4 # Paths to the model checkpoints.
models = [oatomobile.baselines.torch.ImitativeModel() for _ in range(4)]
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
# ckpt = "data/model/dim/ckpts/model-0.pt" # Paths to the model checkpoints.
# model = oatomobile.baselines.torch.ImitativeModel()
# model.load_state_dict(torch.load(ckpt))

# Initializes a CARLA environment.
environment = CARLAEnv(town="Town04")
# Makes an initial observation.
observation = environment.reset()
done = False

agent = oatomobile.baselines.torch.RIPAgent(
  environment=environment,
  models=models,
  algorithm="WCM",
  )

while not done:
  action = agent.act(observation)
  observation, reward, done, info = environment.step(action)
  # Renders interactive display.
  environment.render(mode="human")

# # Book-keeping: closes
environment.close()
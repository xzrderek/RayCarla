# Ensemble Imitation-learners.
from dotenv import load_dotenv
load_dotenv()

import torch
import oatomobile.baselines.torch
import oatomobile
from oatomobile.envs import CARLAEnv

town = "Town02"
ckpts = [ \
  "data-oatml/model/dim/ckpts/model-16.pt",
  "data-oatml/model/dim/ckpts/model-12.pt",
  "data-oatml/model/dim/ckpts/model-8.pt",
  "data-oatml/model/dim/ckpts/model-4.pt", 
  ] # Paths to the model checkpoints.
models = [oatomobile.baselines.torch.ImitativeModel() for _ in range(4)]

for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))

# Initializes a CARLA environment.
environment = CARLAEnv(town=town)
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
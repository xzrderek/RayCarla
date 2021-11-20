# Imitation-learners.

import torch
from torch.utils.tensorboard import SummaryWriter

import oatomobile.baselines.torch
import oatomobile
from oatomobile.envs import CARLAEnv

from absl import app
from absl import flags
from absl import logging

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="model",
    default="data-oatml/model-200/dim/ckpts/model-16.pt",
    help="The name of the model checkpoint.",
)
flags.DEFINE_string(
    name="town",
    default="Town05",
    help="The name of the town for validation.",
)

def main(argv):
  # Debugging purposes.
  # logging.debug(argv)
  # logging.debug(FLAGS)

  # Parses command line arguments.
  ckpt = FLAGS.model
  town_name = FLAGS.town

  model = oatomobile.baselines.torch.ImitativeModel()
  model.load_state_dict(torch.load(ckpt))
  print(model)

  # Initializes a CARLA environment.
  environment = CARLAEnv(town=town_name)
  # Makes an initial observation.
  observation = environment.reset()
  done = False

  agent = oatomobile.baselines.torch.DIMAgent(
    environment=environment,
    model=model,
    )

  while not done:
    action = agent.act(observation)
    observation, reward, done, info = environment.step(action)
    # Renders interactive display.
    environment.render(mode="none")

  # # Book-keeping: closes
  environment.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("model")
  flags.mark_flag_as_required("town")
  app.run(main)

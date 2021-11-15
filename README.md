# RayCarla: Carla on Ray for Autonomous Driving Research.

  **[Overview](#overview)**
| **[Development](#Development)**
| **[Experiments](#Experiments)**

RayCarla is a project of using Carla on Ray for research in Reinforcement Learning, Imitation Learning and Interpretability for Autonomous Driving.

## Overview

If you just want to get started using RayCarla quickly, the first thing to know about the framework is that we wrap [CARLA] towns and scenarios in OpenAI [gym]s:

```python
import torch
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
    default="experiment/data/model/dim/ckpts/model-640.pt",
    help="The name of the model checkpoint.",
)
flags.DEFINE_string(
    name="town",
    default="Town05",
    help="The name of the town for validation.",
)

def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  ckpt = FLAGS.model
  town_name = FLAGS.town

  model = oatomobile.baselines.torch.ImitativeModel()
  model.load_state_dict(torch.load(ckpt))

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
    environment.render(mode="human")

  # # Book-keeping: closes
  environment.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("model")
  flags.mark_flag_as_required("town")
  app.run(main)
```


## Development

1. Install opengl and cuda
    ```bash
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    sudo apt install nvidia-cuda-toolkit
    ```
2. Install pyenv
    ```bash
    curl https://pyenv.run | bash
    ```
3. Add following to ~/.bashrc
    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)" 
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```
4. Restart your shell
    ```bash
    exec $SHELL
    ```
5. Install virtualenv & setuptools
    ```bash
    pip install virtualenv
    pip install --upgrade pip setuptools
    ```
6. Clone code and setup venv
    ```bash
    git clone https://github.com/xzrderek/RayCarla
    cd RayCarla
    virtualenv venv
    source venv/bin/activate
    ```
7. Install Python depdendencies
    ```bash
    pip install -r requirements.txt
    ```
8. Install Carla 
    ```bash
    export CARLA_ROOT=~/carla
    mkdir -p $CARLA_ROOT
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
    tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT
    python -m easy_install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg 
    ```
9. Install Bazel (optional)
    ```bash
    sudo apt install curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
    sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    sudo apt update && sudo apt install bazel
    ```
10. Follow [experiment/experiments.ipynb](https://github.com/xzrderek/RayCarla/blob/main/experiment/experiments.ipynb) notebook to enjoy!

import os
from typing import Mapping

import ray
import ray.train as train
from ray.train import Trainer, TorchConfig
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from absl import app
from absl import flags
from absl import logging

from oatomobile.baselines.torch.dim.model import ImitativeModel
from oatomobile.datasets.carla import CARLADataset

from oatomobile.torch import types
from oatomobile.torch.loggers import TensorBoardLogger
from oatomobile.torch.savers import Checkpointer

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="dataset_dir",
    default=None,
    help="The full path to the processed dataset.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="The full path to the output directory (for logs, ckpts).",
)
flags.DEFINE_integer(
    name="batch_size",
    default=512,
    help="The batch size used for training the neural network.",
)
flags.DEFINE_integer(
    name="num_epochs",
    default=None,
    help="The number of training epochs for the neural network.",
)
flags.DEFINE_integer(
    name="save_model_frequency",
    default=4,
    help="The number epochs between saves of the model.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=1e-3,
    help="The ADAM learning rate.",
)
flags.DEFINE_integer(
    name="num_timesteps_to_keep",
    default=4,
    help="The numbers of time-steps to keep from the target, with downsampling.",
)
flags.DEFINE_float(
    name="weight_decay",
    default=0.0,
    help="The L2 penalty (regularization) coefficient.",
)
flags.DEFINE_bool(
    name="clip_gradients",
    default=False,
    help="If True it clips the gradients norm to 1.0.",
)
flags.DEFINE_integer(
    name="num_workers",
    default=4,
    help="The numbers of ray workers.",
)
flags.DEFINE_string(
    name="ray_address",
    default="127.0.0.1",
    help="The address of ray cluster.",
)

def train_func(config):
    data_size = config.get("data_size", 1000)
    val_size = config.get("val_size", 400)
    batch_size = config.get("batch_size", 32)
    hidden_size = config.get("hidden_size", 1)
    lr = config.get("lr", 1e-2)
    epochs = config.get("epochs", 3)

    train_dataset = LinearDataset(2, 5, size=data_size)
    val_dataset = LinearDataset(2, 5, size=val_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(train_dataset))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(val_dataset))

    model = nn.Linear(1, hidden_size)
    model = DistributedDataParallel(model)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    results = []

    for _ in range(epochs):
        train_epoch(train_loader, model, loss_fn, optimizer)
        result = validate_epoch(validation_loader, model, loss_fn)
        train.report(**result)
        results.append(result)

    return results


def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  dataset_dir = FLAGS.dataset_dir
  output_dir = FLAGS.output_dir
  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  learning_rate = FLAGS.learning_rate
  save_model_frequency = FLAGS.save_model_frequency
  num_timesteps_to_keep = FLAGS.num_timesteps_to_keep
  weight_decay = FLAGS.weight_decay
  clip_gradients = FLAGS.clip_gradients
  noise_level = 1e-2
  num_workers=FLAGS.num_workers

  ray.init(num_cpus=2)
  # ray.init(address=FLAGS.ray_address)

  trainer = Trainer(backend="torch", num_workers=num_workers)
  config = {"lr": learning_rate, "hidden_size": 1, "batch_size": batch_size, "epochs": num_epochs}
  trainer.start()
  results = trainer.run(
        train_func,
        config,
        callbacks=[JsonLoggerCallback(),
                   TBXLoggerCallback()])
  trainer.shutdown()

  print(results)
  


if __name__ == "__main__":
  flags.mark_flag_as_required("dataset_dir")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("num_epochs")
  app.run(main)

# %%
import os
import sys
from os.path import dirname
proj_path = dirname(os.getcwd())
sys.path.append(proj_path)

from typing import Mapping

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

import numpy as np
import ray.train as train
from ray.train import Trainer, TorchConfig
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import ray


# %%

# %%
def train_func(config):
    dataset_dir = "data-oatml/processed" # "data/examples"
    output_dir = "data-oatml/models" # "data/models"
    batch_size = 512
    num_epochs = 20
    save_model_frequency = 4
    num_timesteps_to_keep = 4
    weight_decay = 0.0
    clip_gradients = False
    noise_level = 1e-2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
    device = "cpu"

    # breakpoint()
    
    lr = config.get("lr", 1e-2)
    epochs = config.get("epochs", num_epochs)
    weight_decay = config.get("weight_decay", weight_decay)

    # # Creates the necessary output directory.
    # os.makedirs(output_dir, exist_ok=True)
    # log_dir = os.path.join(output_dir, "logs")
    # os.makedirs(log_dir, exist_ok=True)
    # ckpt_dir = os.path.join(output_dir, "ckpts")
    # os.makedirs(ckpt_dir, exist_ok=True)


    # Initializes the model and its optimizer.
    output_shape = [num_timesteps_to_keep, 2]
    model0 = ImitativeModel(output_shape=output_shape).to(device)
    model = DistributedDataParallel(model0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # writer = TensorBoardLogger(log_dir=log_dir)
    # checkpointer = Checkpointer(model=model, ckpt_dir=ckpt_dir)

    def transform(batch: Mapping[str, types.Array]) -> Mapping[str, torch.Tensor]:
        """Preprocesses a batch for the model. 

        Args:
        batch: (keyword arguments) The raw batch variables.

        Returns:
        The processed batch.
        """
        # Sends tensors to `device`.
        batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
        # Preprocesses batch for the model.
        batch = model0.transform(batch)
        return batch

    # Setups the dataset and the dataloader.
    modalities = (
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "player_future",
        "velocity",
    )
    dataset_train = CARLADataset.as_torch(
        dataset_dir=os.path.join(dataset_dir, "train"),
        modalities=modalities,
    )
    # TODO: shuffle=True,
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset_train),
    )
    dataset_val = CARLADataset.as_torch(
        dataset_dir=os.path.join(dataset_dir, "val"),
        modalities=modalities,
    )
    # TODO: shuffle=True,
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size * 5,
        sampler=DistributedSampler(dataset_val),
    )

    # Theoretical limit of NLL.
    nll_limit = -torch.sum(  # pylint: disable=no-member
        D.MultivariateNormal(
            loc=torch.zeros(output_shape[-2] * output_shape[-1]),  # pylint: disable=no-member
            scale_tril=torch.eye(output_shape[-2] * output_shape[-1]) *  # pylint: disable=no-member
            noise_level,  # pylint: disable=no-member
        ).log_prob(torch.zeros(output_shape[-2] * output_shape[-1])))  # pylint: disable=no-member

    def train_step(
        model: ImitativeModel,
        optimizer: optim.Optimizer,
        batch: Mapping[str, torch.Tensor],
        clip: bool = False,
    ) -> torch.Tensor:
        """Performs a single gradient-descent optimisation step."""
        # Resets optimizer's gradients.
        optimizer.zero_grad()

        # Perturb target.
        y = torch.normal(  # pylint: disable=no-member
            mean=batch["player_future"][..., :2],
            std=torch.ones_like(batch["player_future"][..., :2]) * noise_level,  # pylint: disable=no-member
        )

        # Forward pass from the model.
        z, _ = model0._params(
            velocity=batch["velocity"],
            visual_features=batch["visual_features"],
            is_at_traffic_light=batch["is_at_traffic_light"],
            traffic_light_state=batch["traffic_light_state"],
        )
        _, log_prob, logabsdet = model0._decoder._inverse(y=y, z=z)

        # Calculates loss (NLL).
        loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

        # Backward pass.
        loss.backward()

        # Clips gradients norm.
        if clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # Performs a gradient descent step.
        optimizer.step()

        return loss

    def train_epoch(
        model: ImitativeModel,
        optimizer: optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on `dataloader`."""
        model.train()
        loss = 0.0
        for batch in dataloader:
            # Prepares the batch.
            batch = transform(batch)
            # Performs a gradien-descent step.
            loss += train_step(model, optimizer, batch, clip=clip_gradients)
        return loss / len(dataloader)

    def evaluate_step(
        model: ImitativeModel,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        # Forward pass from the model.
        z, _ = model0._params(
            velocity=batch["velocity"],
            visual_features=batch["visual_features"],
            is_at_traffic_light=batch["is_at_traffic_light"],
            traffic_light_state=batch["traffic_light_state"],
        )
        _, log_prob, logabsdet = model0._decoder._inverse(
            y=batch["player_future"][..., :2],
            z=z,
        )

        # Calculates loss (NLL).
        loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

        print(loss)
        return loss

    def evaluate_epoch(
        model: ImitativeModel,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader."""
        model.eval()
        loss = 0.0
        for batch in dataloader:
            # Prepares the batch.
            batch = transform(batch)
            # Accumulates loss in dataset.
            with torch.no_grad():
                loss += evaluate_step(model, batch)
        loss /= len(dataloader)
        result = {"model": model.state_dict(), "loss": loss}
        return result


    results = []

    for epoch in range(epochs):
        # Trains model on whole training dataset, and writes on `TensorBoard`.
        loss_train = train_epoch(model, optimizer, dataloader_train)
        # write(model, dataloader_train, writer, "train", loss_train, epoch)

        # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
        result = evaluate_epoch(model, dataloader_val)
        # write(model, dataloader_val, writer, "val", loss_val, epoch)

        print(result)
        
        # Checkpoints model weights.
        if epoch % save_model_frequency == 0:
            train.save_checkpoint(epoch=epoch, model=model.module)

        # train_epoch(train_loader, model, loss_fn, optimizer)
        # result = validate_epoch(validation_loader, model, loss_fn)
        train.report(**result)
        results.append(result)

    return results


# %%
smoke_test = True
address = "127.0.0.1"
num_workers = 4

def train_carla():
    trainer = Trainer(TorchConfig(backend="gloo"), num_workers=num_workers, use_gpu=True)
    learning_rate = 1e-3
    config = {"lr": learning_rate}
    trainer.start()
    results = trainer.run(
        train_func,
        config,
        callbacks=[JsonLoggerCallback(),
                   TBXLoggerCallback()])
    trainer.shutdown()

    # print(results)
    return results


if __name__ == "__main__":
    # Start Ray Cluster
    if smoke_test:
        ray.init(num_cpus=32, num_gpus=4) #, local_mode=True)
    else:
        ray.init(address=address)
    # Train carla
    train_carla()

# %%




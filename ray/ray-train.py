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

    # breakpoint()

    dataset_dir = config.get("dataset_dir", "data/examples")
    output_dir = config.get("output_dir", "data/model")
    save_model_frequency = config.get("save_model_frequency", 4)
    num_timesteps_to_keep = config.get("num_timesteps_to_keep", 4)
    
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 512)
    
    weight_decay = config.get("weight_decay", 0.0)
    clip_gradients = config.get("clip_gradients", False)
    noise_level = config.get("noise_level", 1e-2)

    device = torch.device(f"cuda:{train.local_rank()}" if
                  torch.cuda.is_available() else "cpu")
    
    # breakpoint()

    torch.cuda.set_device(device)
        
    # # Creates the necessary output directory.
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # ckpt_dir = os.path.join(output_dir, "ckpts")
    # os.makedirs(ckpt_dir, exist_ok=True)


    # Initializes the model and its optimizer.
    output_shape = [num_timesteps_to_keep, 2]
    model0 = ImitativeModel(output_shape=output_shape).to(device)
    model = DistributedDataParallel(model0, 
                                    device_ids=[train.local_rank()] if torch.cuda.is_available() else None)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    writer = TensorBoardLogger(log_dir=log_dir)
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
        num_workers=50,
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
        num_workers=50,
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

        print("training: {}".format(loss))

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
        loss = loss / len(dataloader)
        return loss
        # result = {"model": model.state_dict(), "loss": loss}
        # return result

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

        print("evaluation: {}".format(loss))
        
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
        return loss
        # result = {"model": model.state_dict(), "loss": loss}
        # return result

    def report(
        model: ImitativeModel,
        dataloader: torch.utils.data.DataLoader,
        writer: TensorBoardLogger,
        split: str,
        loss: torch.Tensor,
        epoch: int,
    ) -> None:
        """Visualises model performance on `TensorBoard`."""
        # Gets a sample from the dataset.
        batch = next(iter(dataloader))
        # Prepares the batch.
        batch = transform(batch)
        # Turns off gradients for model parameters.
        for params in model.parameters():
            params.requires_grad = False
            # Generates predictions.
        predictions = model(num_steps=20, **batch)
        # Turns on gradients for model parameters.
        for params in model.parameters():
            params.requires_grad = True
        # Logs on `TensorBoard`.
        writer.log(
            split=split,
            loss=loss.detach().cpu().numpy().item(),
            overhead_features=batch["visual_features"].detach().cpu().numpy()[:8],
            predictions=predictions.detach().cpu().numpy()[:8],
            ground_truth=batch["player_future"].detach().cpu().numpy()[:8],
            global_step=epoch,
        )


    results = []

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))

        # Trains model on whole training dataset, and writes on `TensorBoard`.
        loss_train = train_epoch(model, optimizer, dataloader_train)

        # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
        loss_val = evaluate_epoch(model, dataloader_val)
        
        # if train.world_rank() == 0:
        report(model, dataloader_train, writer, "train", loss_train, epoch)
        report(model, dataloader_val, writer, "val", loss_val, epoch)
        # Checkpoints model weights.
        if epoch % save_model_frequency == 0:
            train.save_checkpoint(epoch=epoch, model=model.module)

        results.append(loss_val)

    return results


# %%

def train_carla(args):
    trainer = Trainer(backend="torch", num_workers=args.num_workers, use_gpu=args.use_gpu)
    config = {"dataset_dir" : args.dataset_dir, 
              "output_dir" : args.output_dir,
              "epochs" : args.epochs}
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPUs for Ray cluster.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs for Ray cluster.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for training.")
    parser.add_argument(
        "--cluster",
        type=bool,
        default=False,
        help="Connect to existent cluster.")
    parser.add_argument(
        "--address",
        required=False,
        default="127.0.0.1",
        type=str,
        help="The address of the exsitent Ray cluster.")
    parser.add_argument(
        "--dataset_dir",
        default="data/examples",
        type=str,
        help="The dataset directory for training.")
    parser.add_argument(
        "--output_dir",
        default="data/model",
        type=str,
        help="The output directory for training.")
    parser.add_argument(
        "--save_model_frequency",
        type=int,
        default=4,
        help="The frequency of saving model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="The number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="The batch size of training.")
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=True,
        help="Whether to use GPU.")
        
    args, _ = parser.parse_known_args()
    # Start Ray Cluster
    if not args.cluster:
        ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus) #, local_mode=True)
    else:
        ray.init(address=args.address)
    # Train carla
    train_carla(args)

# %%




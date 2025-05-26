"""Training"""

import os
from typing import Optional

import hydra
import torch
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from pvnet import utils
from tqdm import tqdm
from ocf_data_sampler.torch_datasets.sample.base import copy_batch_to_device

from pvnet_summation.data.datamodule import SavedPredictionDataModule

log = utils.get_logger(__name__)

torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_monitor_loss(output_quantiles):
    """Return the desired metric to monitor based on whether quantile regression is being used.

    The adds the option to use something like:
        monitor: "${resolve_monitor_loss:${model.output_quantiles}}"

    in early stopping and model checkpoint callbacks so the callbacks config does not need to be
    modified depending on whether quantile regression is being used or not.
    """
    if output_quantiles is None:
        return "MAE/val"
    else:
        return "quantile_loss/val"


OmegaConf.register_new_resolver("resolve_monitor_loss", resolve_monitor_loss)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.to(device)

    # Presave batches
    if config.get("presave_pvnet_outputs", False):
        save_dir = (
            f"{config.datamodule.sample_dir}/"
            f"{config.model.model_name}/"
            f"{config.model.model_version}"
        )

        if os.path.isdir(save_dir):
            log.info(
                f"PVNet output directory already exists: {save_dir}\n"
                "Skipping saving new outputs. The existing saved outputs will be loaded."
            )

        else:
            log.info(f"Saving PVNet outputs to {save_dir}")

            os.makedirs(f"{save_dir}/train")
            os.makedirs(f"{save_dir}/val")

            # Set batch size to None so batching is skipped
            datamodule.batch_size = None

            for dataloader_func, split in [
                (datamodule.train_dataloader, "train"),
                (datamodule.val_dataloader, "val"),
            ]:
                log.info(f"Saving {split} outputs")
                dataloader = dataloader_func(shuffle=False)

                for i, conc_sample_dict in tqdm(enumerate(dataloader)):
                    # Run PVNet inputs though model and add to sample
                    x = [copy_batch_to_device(conc_sample_dict["pvnet_inputs"], device)]
                    conc_sample_dict["pvnet_outputs"] = model.predict_pvnet_batch(x)[0].cpu()

                    # No longer need the PVNet inputs, and they are big, so remove them from sample
                    del conc_sample_dict["pvnet_inputs"]

                    # Clip the inputs specifically for this summation model
                    for key in ["national_targets", "times"]:
                        conc_sample_dict[key] = conc_sample_dict[key][: model.forecast_len]

                    # Save pvnet prediction sample
                    sample_path = f"{save_dir}/{split}/{i:06}.pt"
                    torch.save(conc_sample_dict, sample_path)

        datamodule = SavedPredictionDataModule(
            sample_dir=save_dir,
            batch_size=config.datamodule.batch_size,
            num_workers=config.datamodule.num_workers,
            prefetch_factor=config.datamodule.prefetch_factor,
        )

    # Init lightning loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    # - this makes it easy to push the model to huggingface
    use_wandb_logger = False
    for logger in loggers:
        log.info(f"{logger}")
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    if use_wandb_logger:
        for callback in callbacks:
            log.info(f"{callback}")
            if isinstance(callback, ModelCheckpoint):
                # Need to call the .experiment property to initialise the logger
                wandb_logger.experiment
                callback.dirpath = "/".join(
                    callback.dirpath.split("/")[:-1] + [wandb_logger.version]
                )
                # Also save model config here - this makes for easy model push to huggingface
                os.makedirs(callback.dirpath, exist_ok=True)
                OmegaConf.save(config.model, f"{callback.dirpath}/model_config.yaml")
                break

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)

    # Validate after end - useful if using stochastic weight averaging
    trainer.validate(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        loggers=loggers,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]

"""Base model for all PVNet submodels"""

import logging
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import wandb
import pvnet
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.models.base_model import PVNetModelHubMixin
from pvnet.models.utils import (
    MetricAccumulator,
    PredAccumulator,
)
from pvnet.optimizers import AbstractOptimizer

from pvnet_summation.utils import plot_forecasts

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


class BaseModel(PVNetBaseModel):
    """Abtstract base class for PVNet summation submodels"""

    def __init__(
        self,
        model_name: str,
        model_version: Optional[str],
        num_locations: int = 317,
        optimizer: AbstractOptimizer = pvnet.optimizers.Adam(),
        output_quantiles: Optional[list[float]] = None,
    ):
        """Abtstract base class for PVNet summation submodels.

        Args:
            model_name: Model path either locally or on huggingface.
            model_version: Model version if using huggingface. Set to None if using local.
            num_locations: The number of regional GSP locations.
            optimizer (AbstractOptimizer): Optimizer
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
        """
        pl.LightningModule.__init__(self)
        PVNetModelHubMixin.__init__(self)

        self.pvnet_model_name = model_name
        self.pvnet_model_version = model_version
        self.num_locations = num_locations

        self.pvnet_model = PVNetBaseModel.from_pretrained(
            model_id=model_name,
            revision=model_version,
        )
        self.pvnet_model.requires_grad_(False)

        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        self.forecast_minutes = self.pvnet_model.forecast_minutes
        self.output_quantiles = output_quantiles

        # Number of timestemps for 30 minutely data
        self.forecast_len = self.forecast_minutes // 30

        self._accumulated_metrics = MetricAccumulator()
        self._accumulated_y = PredAccumulator()
        self._accumulated_y_hat = PredAccumulator()
        self._accumulated_y_sum = PredAccumulator()
        self._accumulated_times = PredAccumulator()
        self._horizon_maes = MetricAccumulator()
        self._gsp_horizon_maes = MetricAccumulator()

        self.use_quantile_regression = self.output_quantiles is not None

        if self.use_quantile_regression:
            self.num_output_features = self.forecast_len * len(self.output_quantiles)
        else:
            self.num_output_features = self.forecast_len

        if self.pvnet_model.use_quantile_regression:
            self.pvnet_output_shape = (
                num_locations,
                self.pvnet_model.forecast_len,
                len(self.pvnet_model.output_quantiles),
            )
        else:
            self.pvnet_output_shape = (num_locations, self.pvnet_model.forecast_len)

        self.use_weighted_loss = False

    def predict_pvnet_batch(self, batch):
        """Use PVNet model to create predictions for batch"""
        gsp_batches = []
        for sample in batch:
            preds = self.pvnet_model(sample)
            gsp_batches += [preds]
        return torch.stack(gsp_batches)

    def sum_of_gsps(self, x):
        """Compute the sume of the GSP-level predictions"""
        if self.pvnet_model.use_quantile_regression:
            y_hat = self.pvnet_model._quantiles_to_prediction(x["pvnet_outputs"])
        else:
            y_hat = x["pvnet_outputs"]

        return (y_hat * x["effective_capacity"]).sum(dim=1)

    def _training_accumulate_log(self, batch_idx, losses, y_hat, y, y_sum, times):
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

        losses = {k: v.detach().cpu() for k, v in losses.items()}
        y_hat = y_hat.detach().cpu()

        self._accumulated_metrics.append(losses)
        self._accumulated_y_hat.append(y_hat)
        self._accumulated_y.append(y)
        self._accumulated_y_sum.append(y_sum)
        self._accumulated_times.append(times)

        if not self.trainer.fit_loop._should_accumulate():
            losses = self._accumulated_metrics.flush()
            y_hat = self._accumulated_y_hat.flush()
            y = self._accumulated_y.flush()
            y_sum = self._accumulated_y_sum.flush()
            times = self._accumulated_times.flush()

            self.log_dict(
                losses,
                on_step=True,
                on_epoch=True,
            )

            # Number of accumulated grad batches
            grad_batch_num = (batch_idx + 1) / self.trainer.accumulate_grad_batches

            # We only create the figure every 8 log steps
            # This was reduced as it was creating figures too often
            if grad_batch_num % (8 * self.trainer.log_every_n_steps) == 0:
                fig = plot_forecasts(
                    y,
                    y_hat,
                    times,
                    batch_idx,
                    quantiles=self.output_quantiles,
                    y_sum=y_sum,
                )
                fig.savefig("latest_logged_train_batch.png")

    def training_step(self, batch, batch_idx):
        """Run training step"""

        y_hat = self.forward(batch)
        y = batch["national_targets"]
        times = batch["times"]
        y_sum = self.sum_of_gsps(batch)

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log(batch_idx, losses, y_hat, y, y_sum, times)

        if self.use_quantile_regression:
            opt_target = losses["quantile_loss/train"]
        else:
            opt_target = losses["MAE/train"]

        return opt_target

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""

        y_hat = self.forward(batch)
        y = batch["national_targets"]
        times = batch["times"]
        y_sum = self.sum_of_gsps(batch)

        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        # Store these to make horizon accuracy plot
        self._horizon_maes.append(
            {i: losses[f"MAE_horizon/step_{i:03}"].cpu().numpy() for i in range(self.forecast_len)}
        )

        logged_losses = {f"{k}/val": v for k, v in losses.items()}

        # Add losses for sum of GSP predictions
        logged_losses.update(
            {
                "MSE/val_gsp_sum": F.mse_loss(y_sum, y),
                "MAE/val_gsp_sum": F.l1_loss(y_sum, y),
            }
        )

        gsp_sum_mae_each_step = torch.mean(torch.abs(y_sum - y), dim=0)

        self._gsp_horizon_maes.append(
            {i: gsp_sum_mae_each_step[i].cpu().numpy() for i in range(self.forecast_len)}
        )

        self.log_dict(
            logged_losses,
            on_step=False,
            on_epoch=True,
        )

        accum_batch_num = batch_idx // self.trainer.accumulate_grad_batches

        if accum_batch_num in [0, 1]:
            # Store these temporarily under self
            if not hasattr(self, "_val_y_hats"):
                self._val_y_hats = PredAccumulator()
                self._val_y = PredAccumulator()
                self._val_y_sum = PredAccumulator()
                self._val_times = PredAccumulator()

            self._val_y_hats.append(y_hat)
            self._val_y.append(y)
            self._val_y_sum.append(y_sum)
            self._val_times.append(times)

            # if batch had accumulated
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                y_hat = self._val_y_hats.flush()
                y = self._val_y.flush()
                y_sum = self._val_y_sum.flush()
                times = self._val_times.flush()

                fig = plot_forecasts(
                    y,
                    y_hat,
                    times,
                    batch_idx,
                    quantiles=self.output_quantiles,
                    y_sum=y_sum,
                )

                self.logger.experiment.log(
                    {
                        f"val_forecast_samples/batch_idx_{accum_batch_num}": wandb.Image(fig),
                    }
                )
                del self._val_y_hats
                del self._val_y
                del self._val_y_sum
                del self._val_times

        return logged_losses

    def on_validation_epoch_end(self):
        """Run on epoch end"""

        horizon_maes_dict = self._horizon_maes.flush()
        gsp_sum_horizon_maes_dict = self._gsp_horizon_maes.flush()

        # Create the horizon accuracy curve
        if isinstance(self.logger, pl.loggers.WandbLogger):
            per_step_losses = [[i, horizon_maes_dict[i]] for i in range(self.forecast_len)]
            try:
                table = wandb.Table(data=per_step_losses, columns=["horizon_step", "MAE"])
                wandb.log(
                    {
                        "horizon_loss_curve": wandb.plot.line(
                            table, "horizon_step", "MAE", title="Horizon loss curve"
                        )
                    },
                )
            except Exception as e:
                print("Failed to log horizon_loss_curve to wandb")
                print(e)

            per_step_losses = [[i, gsp_sum_horizon_maes_dict[i]] for i in range(self.forecast_len)]
            try:
                table = wandb.Table(data=per_step_losses, columns=["horizon_step", "MAE"])
                wandb.log(
                    {
                        "gsp_sum_horizon_loss_curve": wandb.plot.line(
                            table, "horizon_step", "MAE", title="GSP-sum horizon loss curve"
                        )
                    },
                )
            except Exception as e:
                print("Failed to log horizon_loss_curve to wandb")
                print(e)

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)

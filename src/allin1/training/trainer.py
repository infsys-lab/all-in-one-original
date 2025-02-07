import warnings
import librosa
import numpy as np
import torch.nn.functional as F
import torch

from typing import Dict, Union
from lightning import LightningModule
from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation
from numpy.typing import NDArray
from sklearn.metrics import f1_score, accuracy_score
from timm.optim.optim_factory import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from ..models import AllInOne
from ..typings import AllInOneOutput, AllInOnePrediction
from ..config import Config
from .helpers import local_maxima

# Ignore specific warnings from libraries
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='Not enough beat annotations')
warnings.filterwarnings('ignore', category=UserWarning, message='The epoch parameter')
warnings.filterwarnings('ignore', category=UserWarning, message='no annotated tempo strengths given')


class AllInOneTrainer(LightningModule):
    scheduler: Scheduler

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.model == 'allinone':
            self.model = AllInOne(cfg)
        else:
            raise NotImplementedError(f'Unknown model: {cfg.model}')

        self.lr = cfg.lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self,
            opt=self.cfg.optimizer,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        if self.cfg.sched is not None:
            self.scheduler, _ = create_scheduler(self.cfg, optimizer)

        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        batch_size = batch['spec'].shape[0]
        outputs: AllInOneOutput = self(batch['spec'])
        losses = self.compute_losses(outputs, batch, prefix='train/')
        loss = losses.pop('train/loss')
        self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)
        self.log_dict(losses, batch_size=batch_size)

        # Debugging Predictions for Each Epoch
        predictions = self.compute_predictions(outputs, mask=batch['mask'])
        print(f"\nEpoch {self.current_epoch+1} Training Predictions:")
        print(f"  Function Predictions: {predictions.pred_functions}")
        print(f"  Beat Predictions: {predictions.pred_beats.sum()} beats detected")
        print(f"  Downbeat Predictions: {predictions.pred_downbeats.sum()} downbeats detected")
        print(f"  Section Predictions: {predictions.pred_sections.sum()} sections detected")

        # Log Label Distribution
        unique, counts = np.unique(predictions.pred_functions, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        print(f"  Label Distribution: {label_distribution}")

        return loss

    def evaluation_step(self, batch, batch_idx, prefix=None):
        batch_size = batch['spec'].shape[0]
        outputs: AllInOneOutput = self(batch['spec'])
        losses = self.compute_losses(outputs, batch, prefix)
        predictions = self.compute_predictions(outputs)
        scores = self.compute_metrics(predictions, batch, prefix)

        self.log_dict(losses, sync_dist=True, batch_size=batch_size)
        self.log_dict(scores, sync_dist=True, batch_size=batch_size)

        # Debugging Predictions for Validation & Testing
        print(f"\nEpoch {self.current_epoch+1} {prefix} Predictions:")
        print(f"  Function Predictions: {predictions.pred_functions}")
        print(f"  Beat Predictions: {predictions.pred_beats.sum()} beats detected")
        print(f"  Downbeat Predictions: {predictions.pred_downbeats.sum()} downbeats detected")
        print(f"  Section Predictions: {predictions.pred_sections.sum()} sections detected")

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, prefix='val/')

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, prefix='test/')

    def compute_predictions(self, outputs: AllInOneOutput, mask=None):
        raw_prob_beats = torch.sigmoid(outputs.logits_beat.detach())
        raw_prob_downbeats = torch.sigmoid(outputs.logits_downbeat.detach())
        raw_prob_sections = torch.sigmoid(outputs.logits_section.detach())
        raw_prob_functions = torch.softmax(outputs.logits_function.detach(), dim=1)

        prob_beats, _ = local_maxima(raw_prob_beats, filter_size=self.cfg.min_hops_per_beat + 1)
        prob_downbeats, _ = local_maxima(raw_prob_downbeats, filter_size=4 * self.cfg.min_hops_per_beat + 1)
        prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * self.cfg.min_hops_per_beat + 1)
        prob_functions = raw_prob_functions.cpu().numpy()

        if mask is not None:
            prob_beats *= mask
            prob_downbeats *= mask
            prob_sections *= mask

        pred_beats = prob_beats > self.cfg.threshold_beat
        pred_downbeats = prob_downbeats > self.cfg.threshold_downbeat
        pred_sections = prob_sections > self.cfg.threshold_section
        pred_functions = np.argmax(prob_functions, axis=1)
        if mask is not None:
            pred_functions = np.where(mask.cpu().numpy(), pred_functions, -1)

        pred_beat_times = self.tensor_to_time(pred_beats)
        pred_downbeat_times = self.tensor_to_time(pred_downbeats)
        pred_section_times = self.tensor_to_time(pred_sections)

        return AllInOnePrediction(
            raw_prob_beats=raw_prob_beats,
            raw_prob_downbeats=raw_prob_downbeats,
            raw_prob_sections=raw_prob_sections,
            raw_prob_functions=raw_prob_functions,
            prob_beats=prob_beats,
            prob_downbeats=prob_downbeats,
            prob_sections=prob_sections,
            prob_functions=prob_functions,
            pred_beats=pred_beats,
            pred_downbeats=pred_downbeats,
            pred_sections=pred_sections,
            pred_functions=pred_functions,
            pred_beat_times=pred_beat_times,
            pred_downbeat_times=pred_downbeat_times,
            pred_section_times=pred_section_times,
        )

    def on_fit_end(self):
        print("Training Completed.")
        if self.trainer.is_global_zero and self.trainer.checkpoint_callback.best_model_path:
            print("=> Loading Best Model...")
            self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
            print("Best Model Loaded.")

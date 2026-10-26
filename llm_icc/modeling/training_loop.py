"""Manual training loops (no HF Trainer)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from llm_icc.modeling.early_stopping import EarlyStopping
from llm_icc.modeling.losses import sft_cross_entropy, weighted_distribution_mse


@dataclass
class TrainingResult:
    best_dev_loss: float | None = None
    best_step: int | None = None
    steps: int = 0
    history: list[dict] = field(default_factory=list)


class _BaseTrainer:
    def __init__(self, simulator, train_ds, config, logger, learning_rate: float, epochs: int,
                 seed: int = 0, tracker=None):
        self.sim, self.train_ds, self.cfg, self.log = simulator, train_ds, config, logger
        self.lr, self.epochs, self.seed, self.tracker = learning_rate, epochs, seed, tracker

    def build_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(self.sim.trainable_parameters(), lr=self.lr,
                                 weight_decay=self.cfg.training.weight_decay)

    def build_scheduler(self, optimizer, num_training_steps: int) -> LambdaLR:
        """Linear decay with linear warmup."""
        warm = int(self.cfg.training.warmup_ratio * num_training_steps)

        def lam(step):
            if warm > 0 and step < warm:
                return step / warm
            return max(0.0, (num_training_steps - step) / max(1, num_training_steps - warm))
        return LambdaLR(optimizer, lam)

    def _loader(self, ds, collate, shuffle: bool, epoch: int = 0):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return DataLoader(ds, batch_size=self.cfg.training.batch_size, shuffle=shuffle, generator=g,
                          collate_fn=partial(collate, pad_id=self.sim.pad_id))

    def _loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def _optimizer_steps(self) -> int:
        accum = self.cfg.training.gradient_accumulation_steps
        per_epoch = math.ceil(math.ceil(len(self.train_ds) / self.cfg.training.batch_size) / accum)
        return per_epoch * self.epochs

    def _epoch_hook(self, epoch: int) -> None:
        pass

    def _run(self, collate, on_step=None) -> int:
        """Shared loop: accumulate -> clip -> step -> schedule. `on_step(step)` returns True to stop."""
        t = self.cfg.training
        opt = self.build_optimizer()
        sched = self.build_scheduler(opt, self._optimizer_steps())
        params = list(self.sim.trainable_parameters())
        step = 0
        stop = False
        for epoch in range(self.epochs):
            self._epoch_hook(epoch)
            self.sim.train_mode()
            loader = self._loader(self.train_ds, collate, shuffle=True, epoch=epoch)
            n_batches = len(loader)
            running, count = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                loss = self._loss(batch)
                (loss / t.gradient_accumulation_steps).backward()
                running += float(loss.detach())
                count += 1
                if (i + 1) % t.gradient_accumulation_steps == 0 or i + 1 == n_batches:
                    torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 50 == 0:
                        self.log.info("epoch %d step %d train_loss %.5f lr %.2e",
                                      epoch, step, running / count, sched.get_last_lr()[0])
                        if self.tracker:
                            self.tracker({"train_loss": running / count}, step)
                        running, count = 0.0, 0
                    if on_step is not None and on_step(step):
                        stop = True
                        break
            if stop:
                break
        return step


class SFTTrainer(_BaseTrainer):
    """Stage 1: cross-entropy on one sampled answer token per (item, descriptor)."""

    def __init__(self, simulator, train_ds, config, logger, seed: int = 0, tracker=None):
        s = config.training.sft
        super().__init__(simulator, train_ds, config, logger, s.learning_rate, s.epochs, seed, tracker)

    def _epoch_hook(self, epoch: int) -> None:
        self.train_ds.resample(epoch)

    def _loss(self, batch):
        return sft_cross_entropy(self.sim.answer_logits(batch), batch["labels"].to(self.sim.device))

    def train(self) -> TrainingResult:
        steps = self._run(self.sim.sft_collate)
        return TrainingResult(steps=steps)


class DistributionTrainer(_BaseTrainer):
    """Stage 2: omega-weighted MSE between restricted-softmax option probabilities and targets."""

    def __init__(self, simulator, train_ds, dev_ds, config, logger, checkpoint_dir: Path | None = None,
                 seed: int = 0, tracker=None):
        super().__init__(simulator, train_ds, config, logger, config.training.learning_rate,
                         config.training.epochs, seed, tracker)
        self.dev_ds = dev_ds
        self.checkpoint_dir = checkpoint_dir

    def _loss(self, batch):
        probs = self.sim.option_probabilities(batch)
        d = self.sim.device
        return weighted_distribution_mse(probs, batch["target"].to(d), batch["option_mask"].to(d),
                                         batch["weight"].to(d))

    @torch.no_grad()
    def evaluate(self, ds) -> dict[str, float]:
        """omega-weighted dev MSE (early-stop metric), pooled over the whole set."""
        self.sim.eval_mode()
        d = self.sim.device
        num, den = 0.0, 0.0
        for batch in self._loader(ds, self.sim.distribution_collate, shuffle=False):
            probs = self.sim.option_probabilities(batch)
            w = batch["weight"].to(d).float()
            sq = ((probs.float() - batch["target"].to(d)) ** 2) * batch["option_mask"].to(d).float()
            num += float((w * sq.sum(-1)).sum())
            den += float(w.sum())
        self.sim.train_mode()
        return {"dev_loss": num / den}

    def train(self) -> TrainingResult:
        t = self.cfg.training
        stopper = EarlyStopping(t.early_stopping_patience)
        result = TrainingResult()

        def check(step: int) -> bool:
            m = self.evaluate(self.dev_ds)
            result.history.append({"step": step, **m})
            improved = stopper.update(m["dev_loss"], step, self.sim.get_trainable_state)
            self.log.info("step %d dev_loss %.5f%s", step, m["dev_loss"], " (best)" if improved else "")
            if self.tracker:
                self.tracker(m, step)
            if improved and self.checkpoint_dir is not None:
                self.sim.save(self.checkpoint_dir)
            return stopper.should_stop

        def on_step(step: int) -> bool:
            return step % t.eval_every_steps == 0 and check(step)

        base = self.evaluate(self.dev_ds)  # step 0 = after SFT, so a worse-than-start model is never kept
        stopper.update(base["dev_loss"], 0, self.sim.get_trainable_state)
        result.history.append({"step": 0, **base})
        self.log.info("step 0 dev_loss %.5f", base["dev_loss"])

        result.steps = self._run(self.sim.distribution_collate, on_step)
        if result.steps % t.eval_every_steps != 0:
            check(result.steps)
        if stopper.best_state is not None:
            self.sim.set_trainable_state(stopper.best_state)
        result.best_dev_loss, result.best_step = stopper.best, stopper.best_step
        self.log.info("best dev_loss %.5f at step %d", stopper.best, stopper.best_step)
        return result


class RegressionTrainer(DistributionTrainer):
    """Regression baseline: MSE between the predicted and target item parameter, early-stopped on dev MSE."""

    def _loss(self, batch):
        return F.mse_loss(self.sim.forward(batch), batch["target"].to(self.sim.device))

    @torch.no_grad()
    def evaluate(self, ds) -> dict[str, float]:
        self.sim.eval_mode()
        sq, n = 0.0, 0
        for batch in self._loader(ds, self.sim.distribution_collate, shuffle=False):
            sq += float(((self.sim.forward(batch) - batch["target"].to(self.sim.device)) ** 2).sum())
            n += len(batch["target"])
        self.sim.train_mode()
        return {"dev_loss": sq / n}

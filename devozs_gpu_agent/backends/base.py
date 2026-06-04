"""Backend contract.

A backend knows how to build a trainer for one job and how to publish the final
model. The runner owns dataset loading, checkpoint download/upload, and the
terminal reporting; the backend only differs in *how* training executes.
"""
from abc import ABC, abstractmethod


class TrainingBackend(ABC):
    name = "base"

    @abstractmethod
    def build_trainer(self, job, tokenizer, model, train_ds, eval_ds, output_dir, callbacks):
        """Return a configured (Gaudi)Trainer ready for ``.train()``."""

    @abstractmethod
    def detect_capabilities(self) -> dict:
        """Return a small dict of device specs for registration/heartbeat."""

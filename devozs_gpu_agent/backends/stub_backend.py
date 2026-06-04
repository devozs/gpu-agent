"""No-ML stub backend for local dev.

Exercises the entire pipeline — claim → progress → checkpoint → complete, plus
cooperative stop and resume — without loading a model or touching a GPU. It
returns a tiny fake "trainer" the runner drives the same way as a real one.
"""
import logging
import os
import time

from .base import TrainingBackend

LOGGER = logging.getLogger(__name__)


class _FakeState:
    def __init__(self, max_steps, num_train_epochs):
        self.global_step = 0
        self.max_steps = max_steps
        self.epoch = 0.0
        self.num_train_epochs = num_train_epochs
        self.log_history = []
        self.agent_stopped = False


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


class _FakeArgs:
    def __init__(self, output_dir, num_train_epochs):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs


class StubTrainer:
    """Walks fake steps, invoking the callback so the protocol is fully exercised."""

    def __init__(self, output_dir, steps, epochs, callbacks, resume_step=0):
        self.args = _FakeArgs(output_dir, epochs)
        self.state = _FakeState(steps, epochs)
        self.control = _FakeControl()
        self.callbacks = callbacks
        self.resume_step = resume_step
        for cb in self.callbacks:
            if hasattr(cb, "bind_trainer"):
                cb.bind_trainer(self)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "stub-checkpoint.txt"), "w") as f:
            f.write(f"stub checkpoint at step {self.state.global_step}\n")

    def train(self, resume_from_checkpoint=None):
        start = self.resume_step
        for step in range(start + 1, self.state.max_steps + 1):
            time.sleep(0.3)
            self.state.global_step = step
            self.state.epoch = round(step / max(self.state.max_steps, 1) * self.state.num_train_epochs, 3)
            self.state.log_history.append({"loss": round(2.0 / step, 4)})
            for cb in self.callbacks:
                cb.on_step_end(self.args, self.state, self.control)
            if self.control.should_training_stop:
                LOGGER.info("stub trainer stopping at step %s", step)
                break


class StubBackend(TrainingBackend):
    name = "STUB"

    def load_tokenizer_and_model(self, base_model):
        return None, None

    def build_trainer(self, job, tokenizer, model, train_ds, eval_ds, output_dir, callbacks):
        import json
        hp = json.loads(job.hyperparams or "{}")
        epochs = hp.get("epochs", 3)
        steps = max(int(epochs) * 5, 10)
        return StubTrainer(output_dir, steps, epochs, callbacks)

    def detect_capabilities(self) -> dict:
        return {"backend": "STUB", "note": "no-ML local dev backend"}

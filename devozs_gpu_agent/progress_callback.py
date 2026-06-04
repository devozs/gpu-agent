"""Trainer callback that bridges HuggingFace training to the polling protocol.

On every step it reports epoch/step/loss/percent to management and reads back the
cooperative-stop flag (so a stop request can cross the network). On a stop request
it saves a checkpoint, uploads it, and asks the Trainer to halt at the boundary.
``on_save`` mirrors checkpoints to storage as the Trainer writes them.
"""
import logging

LOGGER = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:
    # Stub-mode dev boxes don't install the ML stack. The stub trainer calls the
    # same on_step_end/on_save hooks, so a plain base class is enough.
    class TrainerCallback:  # type: ignore
        pass


class ManagementProgressCallback(TrainerCallback):
    def __init__(self, client, storage, session_id, report_every=1):
        self.client = client
        self.storage = storage
        self.session_id = session_id
        self.report_every = max(1, report_every)
        self._trainer = None  # set by the backend after Trainer construction

    def bind_trainer(self, trainer):
        self._trainer = trainer

    def _latest_loss(self, state):
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    return float(entry["loss"])
        return None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.report_every != 0:
            return control
        total = max(state.max_steps, 1)
        report = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "totalEpochs": float(args.num_train_epochs),
            "step": int(state.global_step),
            "totalSteps": int(state.max_steps),
            "loss": self._latest_loss(state),
            "percent": int(100 * state.global_step / total),
        }
        try:
            stop_requested = self.client.report_progress(self.session_id, report)
        except Exception:
            LOGGER.warning("progress report failed (non-fatal)", exc_info=True)
            return control

        if stop_requested:
            LOGGER.info("stop requested by management — checkpointing and halting")
            self._checkpoint(args, state)
            control.should_training_stop = True
            control.should_save = False
            # Signal the runner that this was a cooperative stop, not completion.
            setattr(state, "agent_stopped", True)
        return control

    def on_save(self, args, state, control, **kwargs):
        self._checkpoint(args, state)
        return control

    def _checkpoint(self, args, state):
        if self._trainer is None:
            return
        local_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        try:
            self._trainer.save_model(local_dir)
            uri = self.storage.upload_checkpoint(local_dir, state.global_step)
            self.client.report_checkpoint(self.session_id, uri)
            self.client.report_log(self.session_id, "INFO", f"checkpoint at step {state.global_step}")
        except Exception:
            LOGGER.warning("checkpoint save/upload failed", exc_info=True)

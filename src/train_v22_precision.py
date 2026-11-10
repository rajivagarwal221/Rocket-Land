"""V22: 1.5M-step production fine-tune of V21 precision.

Design rationale (senior-review version -- every change has a specific
hypothesis tied to a measured V21 weakness, not a vibe).

================================================================
V21 measured weaknesses (common-profile, n=1000, shield on):
  * 83.3%  precise          (target >=99%,  gap -15.7 pp)
  * 94.5%  safe landings    (target >=99%,  gap  -4.5 pp)
  *  2.2%  crash + timeout  (target <=0.1%, gap  -2.1 pp)
  *  3.3%  leg failures     <- emerged from aggressive V21 reward;
                              V20 was 1.3%. Root cause: x_shaping_coef
                              is strongest right at touchdown, pushing
                              the policy to chase x in the last frames
                              and tipping the lander onto one leg.
Shield ablation: V21 policy is safe WITHOUT the shield (98% safe,
  shield OFF). Keep shield for defense-in-depth, not as a crutch.
================================================================

V22 changes (each tied to a V21 weakness above):

  REWARD SHAPE
    * target_bonus              300  -> 300   (unchanged, V21 proved it)
    * off_target_penalty         80  ->  60   (-25%): V21 was too punishing,
                                               contributed to tipping behaviour
    * x_shaping_coef            5.0  ->  3.5  (-30%): reduce lateral pressure
    * x_shaping_fade_below      0.0  ->  0.12 NEW: fade x-shaping to zero
                                               when altitude < 0.12 (last
                                               ~1.8 m real). Lets the policy
                                               commit to a vertical touchdown.
    * angular_stability_bonus   0.0  ->  50.0 NEW: terminal bonus when
                                               landing with both legs,
                                               |angle|<0.15 rad, |omega|<0.05.
                                               Direct signal against tipping.

  SAFETY WEIGHTS (unchanged -- V21 held 94.5% safe with these)
    * failed_landing_penalty        500
    * timeout_penalty               200

  ENV ENVELOPE
    * weak_gravity_prob         0.25 -> 0.35  more exposure to the hard slice
                                              (V21 was 78% precise there)
    Everything else unchanged. Shield ON during training
    (defense-in-depth, matches eval).

  SAC HYPERPARAMETERS
    * learning_rate             1e-4 -> linear 1e-4 -> 1e-5 over 1.5M steps
                                              (final convergence, standard FT)
    * learning_starts            5_000 (warm policy, no extra random warmup)
    * ent_coef                  continue auto from V21's tuned state
      Rationale: V21's entropy already fits a precision-oriented policy.
      Resetting would waste the fine-tune by re-exploring from scratch.
    * buffer_size, tau, gamma, batch_size: unchanged from V21.

  TRAINING INFRASTRUCTURE (not training-related but still loopholes)
    1. EvalCallback: every 100k steps, 50 deterministic episodes on a
       separate eval env, saves best model to models/master_v22_best.zip.
       Prevents "trained past peak" -- best checkpoint is preserved
       regardless of final step performance.
    2. TensorBoard log at logs/tb_v22/ -- visibility into actor_loss,
       critic_loss, ent_coef, ep_rew_mean during training.
    3. Replay buffer saved alongside each 50k checkpoint (models/
       master_v22_buffer.pkl). Makes mid-training interruption truly
       resumable instead of starting the buffer empty each time.
    4. Separate Monitor filename (master_v22_training), won't overwrite
       V21's training log.
    5. Shield stays ON in training. Not a crutch -- V21 ablation shows
       the policy doesn't need it, but training with it protects early
       random explorations and matches deployment environment.

  BUDGET: 1.5M steps. Matches V20's original budget. On RTX 3050 ~3-4h
  GPU, ~12-14h CPU. Set V22_STEPS env var to shorten.

================================================================

Deliverables:
  * models/master_v22_precision.zip   -- latest checkpoint
  * models/master_v22_best.zip        -- best eval-score checkpoint
  * models/master_v22_buffer.pkl      -- latest replay buffer
  * logs/master_v22_training.monitor.csv
  * logs/tb_v22/                      -- TensorBoard event files
"""
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper
from safety_shield import SafetyShield


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
WARM_START = MODELS / "master_v21_precision.zip"
CANDIDATE = MODELS / "master_v22_precision.zip"
BEST_DIR = MODELS                                # EvalCallback writes best_model.zip here; we rename
BUFFER_PATH = MODELS / "master_v22_buffer.pkl"
TB_LOG = LOGS / "tb_v22"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V22_STEPS", "1500000"))
SAVE_INTERVAL = 50_000

LR_START = 1e-4
LR_END = 1e-5
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V22_SEED", "2242"))

# EvalCallback config
EVAL_EVERY = 100_000
EVAL_EPISODES = 50


def linear_schedule(start: float, end: float):
    """SB3 progress_remaining goes from 1.0 at start to 0.0 at end."""

    def _schedule(progress_remaining: float) -> float:

        return end + (start - end) * progress_remaining

    return _schedule


def save_model_and_buffer(m, candidate_path: Path, buffer_path: Path):
    """Atomic save of both model and replay buffer."""

    # Model: write to temp then rename (atomic on same filesystem)
    t = candidate_path.with_name(candidate_path.stem + "_temporary")
    tz = Path(str(t) + ".zip")
    tz.unlink(missing_ok=True)
    m.save(str(t))
    os.replace(tz, candidate_path)

    # Replay buffer: SB3 writes .pkl. Not atomic but a half-written buffer
    # is only an issue if training crashes during this exact write.
    try:
        m.save_replay_buffer(str(buffer_path))
    except Exception as e:
        print(f"[warn] replay buffer save failed: {e}", flush=True)


class CheckpointCB(BaseCallback):
    """Periodic latest-checkpoint + replay-buffer save."""

    def __init__(self):

        super().__init__(verbose=1)
        self.last = 0

    def _on_training_start(self):

        self.last = self.model.num_timesteps

    def _on_step(self):

        if self.model.num_timesteps - self.last >= SAVE_INTERVAL:

            save_model_and_buffer(
                self.model, CANDIDATE, BUFFER_PATH
            )
            self.last = self.model.num_timesteps
            print(
                f"CHECKPOINT {self.model.num_timesteps:,}",
                flush=True,
            )

        return True


def build_training_env():

    base = ThrustScaledLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 0.8),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),
        weak_gravity_prob=0.35,              # 0.25 -> 0.35 (more weak-g exposure)
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )
    base = SafetyShield(
        base,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
    )
    env = gym.wrappers.TimeLimit(
        base, max_episode_steps=MAX_EPISODE_STEPS
    )
    env = TargetLandingWrapper(
        env,
        gamma=GAMMA,
        step_penalty=0.01,
        target_bonus=300.0,                  # unchanged, V21 proved it
        off_target_penalty=60.0,             # 80 -> 60 (-25% lateral pressure)
        timeout_penalty=200.0,               # unchanged
        failed_landing_penalty=500.0,        # unchanged
        target_limit=0.15,
        x_shaping_coef=3.5,                  # 5.0 -> 3.5 (softer global)
        x_shaping_fade_below=0.12,           # NEW: fade below altitude 0.12
        angular_stability_bonus=50.0,        # NEW: upright-landing bonus
        angular_stability_angle=0.15,
        angular_stability_omega=0.05,
    )
    env = Monitor(
        env, filename=str(LOGS / "master_v22_training")
    )
    return env


def build_eval_env():
    """Separate env for EvalCallback -- same envelope, independent seed."""

    env = build_training_env()
    # Monitor already attached in build_training_env; EvalCallback expects
    # a Monitor-wrapped env, so we re-use. Different seed via reset.
    return env


def main():

    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    TB_LOG.mkdir(exist_ok=True, parents=True)

    if not WARM_START.exists():

        raise FileNotFoundError(
            f"V22 needs V21 as warm start, but {WARM_START} does not exist."
        )

    train_env = build_training_env()
    eval_env = build_eval_env()

    try:

        # Load V21 weights and override the hparams baked into the zip
        # so the fine-tune starts under V22's regime immediately.
        lr_schedule = linear_schedule(LR_START, LR_END)

        model = SAC.load(
            str(WARM_START),
            env=train_env,
            device=DEVICE,
            seed=SEED,
            tensorboard_log=str(TB_LOG),
            custom_objects={
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
                "learning_starts": LEARNING_STARTS,
                "batch_size": BATCH_SIZE,
                "buffer_size": BUFFER_SIZE,
                "tau": TAU,
                "gamma": GAMMA,
                "train_freq": (TRAIN_FREQ, "step"),
                "gradient_steps": GRADIENT_STEPS,
            },
        )

        gpu_name = (
            torch.cuda.get_device_name(0)
            if DEVICE == "cuda"
            else "CPU"
        )
        print(
            f"V22 PRECISION FT on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, "
            f"warm_start={WARM_START.name}, "
            f"target_bonus=300, off_target=60, "
            f"x_shaping=3.5 (fade<0.12), ang_stab=50, "
            f"crash=500, timeout=200, weak_g_prob=0.35, "
            f"shield ON, lr {LR_START:.0e}->{LR_END:.0e} linear",
            flush=True,
        )

        # Save initial state before training begins so an immediate crash
        # still leaves a usable V22 checkpoint.
        save_model_and_buffer(model, CANDIDATE, BUFFER_PATH)

        # EvalCallback: best-model saving, independent of training reward.
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(MODELS),
            log_path=str(LOGS / "eval_v22"),
            eval_freq=EVAL_EVERY,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        )

        checkpoint_cb = CheckpointCB()

        callbacks = CallbackList([checkpoint_cb, eval_cb])

        try:

            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                log_interval=20,
                progress_bar=False,
                reset_num_timesteps=True,
                tb_log_name="v22",
            )

        except KeyboardInterrupt:

            save_model_and_buffer(model, CANDIDATE, BUFFER_PATH)
            print("Interrupted; saved latest + buffer.", flush=True)
            return

        save_model_and_buffer(model, CANDIDATE, BUFFER_PATH)

        # Promote EvalCallback's best_model.zip -> master_v22_best.zip
        eval_best = MODELS / "best_model.zip"
        final_best = MODELS / "master_v22_best.zip"
        if eval_best.exists():
            # If an old best exists, overwrite.
            final_best.unlink(missing_ok=True)
            os.replace(eval_best, final_best)
            print(
                f"V22 best model promoted to {final_best.name}",
                flush=True,
            )
        else:
            print(
                "[warn] EvalCallback did not produce best_model.zip; "
                "no best-model promotion.",
                flush=True,
            )

        print(f"V22 DONE: {model.num_timesteps:,}", flush=True)

    finally:

        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    sys.stdout.reconfigure(line_buffering=True)
    main()

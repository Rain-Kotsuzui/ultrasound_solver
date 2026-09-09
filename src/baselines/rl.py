"""Shared optional Gymnasium environment and SB3 training lifecycle."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from baselines.common import (
    BudgetExhausted, TimeBudgetExhausted, positive_float, positive_int, wrap,
)


class PhaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, problem, horizon=32, action_scale=0.2, reward_scale=100.0,
                 random_reset=False, stage="rl_train"):
        super().__init__()
        self.problem = problem
        self.horizon = positive_int(horizon, "episode_steps")
        self.action_scale = positive_float(action_scale, "action_scale")
        self.reward_scale = positive_float(reward_scale, "reward_scale")
        self.random_reset, self.stage = random_reset, stage
        self.action_space = gym.spaces.Box(-1.0, 1.0, (problem.size,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (2 * problem.size + 3 * len(problem.cfg.targets) + 4,),
            np.float32,
        )
        self.current = None
        self.steps = 0

    def _observation(self):
        value = self.current
        target = np.asarray(self.problem.cfg.targets, dtype=float).reshape(-1, 3)
        target = (target / np.asarray(self.problem.cfg.domain.box_size)).ravel()
        stats = [value.loss / self.reward_scale,
                 value.metrics["target_mean"] / self.reward_scale,
                 value.metrics["background_max"] / self.reward_scale,
                 1 - self.steps / self.horizon]
        observation = np.concatenate(
            [np.sin(value.phases), np.cos(value.phases), target, stats]
        ).astype(np.float32)
        if not np.isfinite(observation).all():
            raise FloatingPointError("RL observation overflow; adjust reward_scale")
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        phases = (self.np_random.uniform(0, 2 * np.pi, self.problem.size)
                  if self.random_reset else self.problem.initial_phases)
        self.steps = 0
        self.current = self.problem.evaluate(phases, stage=self.stage)
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Invalid RL action")
        old_loss = self.current.loss
        phases = wrap(self.current.phases + self.action_scale * np.clip(action, -1, 1))
        self.current = self.problem.evaluate(phases, stage=self.stage)
        self.steps += 1
        reward = (old_loss - self.current.loss) / self.reward_scale
        # The finite horizon is part of the task, not a time-limit truncation.
        terminated = self.steps >= self.horizon
        return self._observation(), reward, terminated, False, {}


def task_fingerprint(problem):
    payload = asdict(problem.cfg)
    for key in ("algorithm", "algorithm_options", "io"):
        payload.pop(key, None)
    for key in ("seed", "max_evaluations", "iterations", "gradient_check"):
        payload["training"].pop(key, None)
    payload["asset_hashes"] = {}
    for obs in problem.cfg.obstacles:
        if obs.get("type") == "mesh":
            payload["asset_hashes"][obs["file"]] = _file_hash(obs["file"])
    if problem.cfg.training.target_field_file:
        name = problem.cfg.training.target_field_file
        payload["asset_hashes"][name] = _file_hash(name)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file_hash(name):
    digest = hashlib.sha256()
    with open(name, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solve_rl(problem, options, method):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback

    cls = {"sac": SAC, "ppo": PPO}[method]
    horizon = positive_int(options.get("episode_steps", 32), "episode_steps")
    eval_steps = positive_int(options.get("evaluation_steps", horizon), "evaluation_steps")
    if eval_steps > horizon:
        raise ValueError("evaluation_steps must not exceed episode_steps")
    action_scale = positive_float(options.get("action_scale", 0.2), "action_scale")
    reward_scale = positive_float(options.get("reward_scale", 100.0), "reward_scale")
    training_steps = positive_int(options.get("total_timesteps", 10000), "total_timesteps")
    run_mode = options.get("run_mode", "train")
    if run_mode not in {"train", "evaluate"}:
        raise ValueError("RL run_mode must be train or evaluate")
    reserved = eval_steps + 1
    if problem.remaining < reserved + (2 if run_mode == "train" else 0):
        raise ValueError("max_evaluations must cover training and reserved policy evaluation")
    checkpoint = Path(options.get("checkpoint", f"outputs/policies/{method}/policy.zip"))
    if checkpoint.suffix != ".zip":
        raise ValueError("RL checkpoint must use .zip extension")
    metadata_path = checkpoint.with_suffix(".json")
    fingerprint = task_fingerprint(problem)
    environment_spec = {"horizon": horizon, "action_scale": action_scale,
                        "reward_scale": reward_scale}
    env = PhaseEnv(problem, horizon, action_scale, reward_scale,
                   bool(options.get("random_reset", False)))
    seed = int(problem.cfg.training.seed)
    start = time.perf_counter()
    trained_evaluations = 0
    reason = "training_steps"
    class WallClockCallback(BaseCallback):
        def _on_step(self):
            nonlocal reason
            if problem.field_evaluations > 0 and problem.time_remaining_seconds <= 0:
                reason = "training_time_budget"
                return False
            return True

    try:
        if run_mode == "evaluate":
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (meta["algorithm"] != method or meta["task_fingerprint"] != fingerprint
                    or meta["environment"] != environment_spec):
                raise ValueError("Policy metadata does not match this task/environment")
            model = cls.load(str(checkpoint), env=env, device=options.get("device", "cpu"))
        else:
            rate = positive_float(options.get("learning_rate", 3e-4), "learning_rate")
            batch = positive_int(options.get("batch_size", 64), "batch_size", 2)
            kwargs = dict(learning_rate=rate, batch_size=batch, gamma=1.0, seed=seed,
                          device=options.get("device", "cpu"), verbose=0)
            if method == "sac":
                kwargs.update(
                    learning_starts=positive_int(options.get("learning_starts", 100),
                                                 "learning_starts", 0),
                    buffer_size=positive_int(options.get("buffer_size", 50000), "buffer_size"),
                    train_freq=1, gradient_steps=1,
                )
            else:
                rollout = positive_int(options.get("n_steps", 128), "n_steps", 2)
                if batch > rollout or rollout % batch:
                    raise ValueError("PPO batch_size must divide n_steps")
                kwargs["n_steps"] = rollout
            model = cls("MlpPolicy", env, **kwargs)
            problem.reserve_evaluations(reserved)
            try:
                model.learn(
                    total_timesteps=training_steps,
                    callback=WallClockCallback(),
                )
            except BudgetExhausted as exc:
                reason = (
                    "training_time_budget"
                    if isinstance(exc, TimeBudgetExhausted)
                    else "training_evaluation_budget"
                )
            finally:
                problem.release_evaluations(reserved)
            trained_evaluations = problem.field_evaluations
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint))
            metadata_path.write_text(json.dumps(
                {"algorithm": method, "task_fingerprint": fingerprint,
                 "environment": environment_spec, "seed": seed,
                 "timesteps": int(model.num_timesteps)}, indent=2), encoding="utf-8")
        train_seconds = time.perf_counter() - start
        test_env = PhaseEnv(problem, horizon, action_scale, reward_scale, stage="rl_evaluate")
        try:
            observation, _ = test_env.reset(seed=seed)
            for _ in range(eval_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, _, _ = test_env.step(action)
                if done:
                    break
            return problem.result(
                test_env.current, reason if run_mode == "train" else "policy_evaluated",
                run_mode=run_mode, checkpoint=str(checkpoint),
                train_field_evaluations=trained_evaluations,
                train_or_load_seconds=train_seconds, timesteps=int(model.num_timesteps),
            )
        finally:
            test_env.close()
    finally:
        env.close()

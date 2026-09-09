import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import run, validate_algorithm, ALGORITHMS
from baselines.common import PhaseProblem, BudgetExhausted
from config import (SimulationConfig, PhysicsConfig, TransducerSpecsConfig,
                    DomainConfig, IOConfig, TrainingConfig)
from physics.transducer_array import TransducerArray


def config(algorithm="adjoint", loss="focal_pressure"):
    return SimulationConfig(
        mode="phase_optimization", algorithm=algorithm,
        physics=PhysicsConfig(), specs=TransducerSpecsConfig(array_n=2),
        domain=DomainConfig(box_size=[0.1]*3, grid_size=[3]*3),
        boundary_conditions={key: "open" for key in ("-x", "+x", "-y", "+y", "-z", "+z")},
        targets=[[0.05]*3], obstacles=[], io=IOConfig(auto_visualize=False),
        training=TrainingConfig(
            initial_phase="zero", loss_type=loss, background_weight=0,
            iterations=8, max_evaluations=100,
            target_radius=0.01, source_exclusion_layers=0, load_basis_to_gpu=False,
            show_loss_curve=False,
        ),
    )


class ToyBasis:
    def __init__(self, cfg):
        self.solver = SimpleNamespace(transducers=TransducerArray(cfg))
        self.num_transducers = 4
        rng = np.random.default_rng(12)
        self.basis = rng.normal(size=(27, 4)) + 1j*rng.normal(size=(27, 4))
        self.calls = 0

    def field(self, phases, return_numpy=True):
        self.calls += 1
        return (self.basis @ np.exp(1j*phases)).reshape((3, 3, 3), order="F")

    def phase_vjp(self, cotangent, phases, return_numpy=True):
        z = self.basis.conj().T @ cotangent.ravel(order="F")
        return np.real(z.conj() * 1j * np.exp(1j*phases))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()

    def tearDown(self):
        self.silence.__exit__(None, None, None)

    def test_analytic_alignment_bound(self):
        cfg = config("response_alignment")
        basis = ToyBasis(cfg)
        problem, result = run(cfg, basis)
        self.assertAlmostEqual(abs(result.final.field[1, 1, 1]),
                               np.sum(abs(basis.basis[13])), places=10)
        self.assertEqual(problem.field_evaluations, 1)
        self.assertEqual(problem.vjp_evaluations, 0)

    def test_scalar_loss_and_gradients(self):
        for loss in ("field_match", "focal_pressure", "focal_contrast"):
            cfg = config(loss=loss)
            cfg.training.background_weight = 0.1
            problem = PhaseProblem(cfg, ToyBasis(cfg))
            phi = np.array([0.2, 0.6, 1.1, 2.3])
            value = problem.evaluate(phi, gradient=True)
            scalar, cotangent = problem.loss_model.evaluate(value.field, False)
            self.assertEqual(scalar, value.loss)
            self.assertIsNone(cotangent)
            for index in range(4):
                delta = np.zeros(4)
                delta[index] = 1e-5
                numerical = (problem.evaluate(phi+delta).loss -
                             problem.evaluate(phi-delta).loss) / 2e-5
                np.testing.assert_allclose(value.gradient[index], numerical, rtol=1e-5, atol=1e-4)

    def test_algorithms_budget_and_reproducibility(self):
        for algorithm in ("adjoint", "geometric", "gabs", "spsa"):
            cfg = config(algorithm)
            cfg.training.max_evaluations = 12
            p1, r1 = run(cfg, ToyBasis(cfg))
            p2, r2 = run(cfg, ToyBasis(cfg))
            self.assertLessEqual(p1.field_evaluations, 12)
            self.assertEqual(p1.field_evaluations, p1.basis.calls)
            np.testing.assert_allclose(r1.final.phases, r2.final.phases)
            self.assertLessEqual(r1.best.loss, r1.final.loss)
            best = [x["best_loss"] for x in r1.history]
            self.assertTrue(np.all(np.diff(best) <= 0))

    def test_adam_and_gradient_diagnostic(self):
        cfg = config()
        cfg.algorithm_options = {
            "optimizer": "adam",
            "gradient_check": True,
            "gradient_check_step": 1e-3,
        }
        problem, result = run(cfg, ToyBasis(cfg))
        self.assertEqual(problem.vjp_evaluations, 9)
        self.assertEqual(problem.field_evaluations, 17)
        self.assertEqual(len(result.metadata["gradient_checks"]), 4)

    def test_budget_invalid_inputs(self):
        cfg = config()
        cfg.training.max_evaluations = 1
        p = PhaseProblem(cfg, ToyBasis(cfg))
        p.evaluate(np.zeros(4))
        with self.assertRaises(BudgetExhausted):
            p.evaluate(np.zeros(4))
        cfg.algorithm = "unknown"
        with self.assertRaises(ValueError):
            validate_algorithm(cfg)

    def test_live_loss_curve_receives_all_evaluations(self):
        cfg = config("spsa")
        cfg.training.show_loss_curve = True
        cfg.training.iterations = 2
        with patch("baselines.loss_curve.LiveLossCurve") as curve_class:
            curve = curve_class.return_value
            problem, result = run(cfg, ToyBasis(cfg))
        self.assertEqual(curve.update.call_count, problem.field_evaluations)
        curve.finalize.assert_called_once_with(result.termination_reason)
        cfg.algorithm = "spsa"
        cfg.algorithm_options = {"typo": 3}
        with self.assertRaises(ValueError):
            validate_algorithm(cfg)

    def test_modes_and_no_legacy_parser(self):
        cfg = config()
        cfg.mode = "same_phase"
        cfg.same_phase_rad = 0.75
        np.testing.assert_allclose(TransducerArray(cfg).phases, 0.75)
        from main import execute
        cfg.mode = "sdf_inverse"
        with self.assertRaises(NotImplementedError):
            execute(cfg)
        raw = yaml.safe_load(Path("src/examples/config.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            raw["mode"] = "unsupported_mode"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises((ValueError, TypeError)):
                SimulationConfig.from_yaml(path)
            raw["mode"] = "phase_optimization"
            raw.setdefault("training", {})["mode"] = "unused"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TypeError):
                SimulationConfig.from_yaml(path)

    @unittest.skipUnless(importlib.util.find_spec("cma"), "optional cma not installed")
    def test_cmaes(self):
        cfg = config("cmaes")
        cfg.training.max_evaluations = 20
        problem, result = run(cfg, ToyBasis(cfg))
        self.assertLessEqual(problem.field_evaluations, 20)
        self.assertTrue(np.isfinite(result.final.loss))

    @unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "optional RL not installed")
    def test_rl_training_loading_and_reward(self):
        from baselines.rl import PhaseEnv
        from stable_baselines3.common.env_checker import check_env
        cfg = config("sac")
        p = PhaseProblem(cfg, ToyBasis(cfg))
        env = PhaseEnv(p, horizon=3, reward_scale=10)
        check_env(env, warn=True)
        env.reset(seed=7)
        initial = env.current.loss
        reward = 0
        for _ in range(3):
            _, r, done, truncated, _ = env.step(np.ones(4, dtype=np.float32)*0.1)
            reward += r
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, (initial-env.current.loss)/10)
        with tempfile.TemporaryDirectory() as directory:
            for method in ("sac", "ppo"):
                cfg = config(method)
                cfg.training.max_evaluations = 40
                cfg.algorithm_options = {
                    "total_timesteps": 8, "episode_steps": 3, "evaluation_steps": 3,
                    "batch_size": 4, "checkpoint": str(Path(directory)/f"{method}.zip"),
                }
                cfg.algorithm_options.update({"learning_starts": 0, "buffer_size": 50}
                                             if method == "sac" else {"n_steps": 8})
                p, result = run(cfg, ToyBasis(cfg))
                self.assertGreater(result.metadata["timesteps"], 0)
                self.assertTrue(Path(cfg.algorithm_options["checkpoint"]).exists())
                cfg.algorithm_options["run_mode"] = "evaluate"
                p, loaded = run(cfg, ToyBasis(cfg))
                self.assertEqual(p.field_evaluations, 4)
                np.testing.assert_allclose(result.final.phases, loaded.final.phases, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

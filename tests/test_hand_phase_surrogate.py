from pathlib import Path
import sys
import unittest

torch = None
try:
    import torch
except ModuleNotFoundError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipIf(torch is None, "torch is not installed")
class HandPhaseSurrogateTests(unittest.TestCase):
    def test_forward_shapes_and_loss_backward(self):
        from algorithm.training.hand_phase_surrogate import (
            AdditiveResidualBEMSurrogate,
            AdditiveResidualSurrogateConfig,
            additive_residual_surrogate_loss,
        )

        config = AdditiveResidualSurrogateConfig(
            num_emitters=4,
            geometry_latent_dim=16,
            target_latent_dim=12,
            hidden_dim=32,
            target_fourier_bands=3,
            geometry_blocks=1,
            target_blocks=1,
            decoder_blocks=1,
        )
        centers = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.00, 0.01, 0.00],
                [0.01, 0.01, 0.00],
            ],
            dtype=torch.float32,
        )
        model = AdditiveResidualBEMSurrogate(
            emitter_centers_m=centers,
            wave_number=2.0 * torch.pi * 40000.0 / 343.0,
            config=config,
        )

        batch_size, target_count = 2, 3
        prediction = model(
            shape_beta=torch.randn(batch_size, 10),
            hand_pose=torch.randn(batch_size, 45),
            root_rotation=torch.randn(batch_size, 3),
            root_translation_m=torch.randn(batch_size, 3) * 0.01,
            target_world_m=torch.rand(batch_size, target_count, 3) * 0.05 + 0.02,
            target_category=torch.zeros(batch_size, target_count, dtype=torch.long),
        )

        self.assertEqual(prediction["response"].shape, (batch_size, target_count, 4))
        self.assertEqual(prediction["delta"].shape, (batch_size, target_count, 4))
        self.assertEqual(prediction["phase"].shape, (batch_size, target_count, 4))
        self.assertTrue(torch.isfinite(prediction["response_real"]).all())
        self.assertTrue(torch.all(prediction["phase"] >= 0.0))
        self.assertTrue(torch.all(prediction["phase"] < 2.0 * torch.pi))

        target = prediction["free_response"].detach()
        losses = additive_residual_surrogate_loss(
            prediction,
            response_real=target.real,
            response_imag=target.imag,
            log_amplitude_weight=0.1,
            phase_weight=0.05,
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

"""Neural surrogate for hand-conditioned BEM target responses.

The model keeps the known free-field response outside the neural network and
learns only the additive correction introduced by the hand:

    H_pred = H_free + dH_pred

The final phase command is derived from the predicted complex response with
``phase = -arg(H_pred)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_batched_targets(target_world_m: Tensor) -> tuple[Tensor, bool]:
    """Return targets as ``(B, T, 3)`` and whether the input was unbatched."""
    if target_world_m.ndim == 2:
        return target_world_m.unsqueeze(0), True
    if target_world_m.ndim != 3 or target_world_m.shape[-1] != 3:
        raise ValueError("target_world_m must have shape (T, 3) or (B, T, 3)")
    return target_world_m, False


def _expand_geometry_feature(feature: Tensor, batch_size: int) -> Tensor:
    if feature.ndim != 2:
        raise ValueError("geometry features must have shape (B, D)")
    if feature.shape[0] == batch_size:
        return feature
    if feature.shape[0] == 1:
        return feature.expand(batch_size, -1)
    raise ValueError(f"geometry batch {feature.shape[0]} does not match target batch {batch_size}")


def compute_point_free_field_response(
    target_world_m: Tensor,
    emitter_centers_m: Tensor,
    wave_number: float | Tensor,
    emitter_amplitudes: Tensor | None = None,
    eps: float = 1.0e-6,
) -> Tensor:
    """Compute a differentiable point-source free-field response.

    Args:
        target_world_m: Target coordinates with shape ``(B, T, 3)`` or ``(T, 3)``.
        emitter_centers_m: Emitter coordinates with shape ``(E, 3)``.
        wave_number: Acoustic wave number ``2*pi*f/c``.
        emitter_amplitudes: Optional per-emitter drive amplitudes ``(E,)``.
        eps: Minimum distance clamp in meters.

    Returns:
        Complex response with shape ``(B, T, E)`` or ``(T, E)`` matching the input
        batching convention.
    """
    targets, unbatched = _as_batched_targets(target_world_m)
    if emitter_centers_m.ndim != 2 or emitter_centers_m.shape[-1] != 3:
        raise ValueError("emitter_centers_m must have shape (E, 3)")

    centers = emitter_centers_m.to(device=targets.device, dtype=targets.dtype)
    diff = targets[:, :, None, :] - centers[None, None, :, :]
    distance = torch.linalg.norm(diff, dim=-1).clamp_min(eps)

    k = torch.as_tensor(wave_number, device=targets.device, dtype=targets.dtype)
    response = torch.exp(1j * k * distance) / (4.0 * torch.pi * distance)
    if emitter_amplitudes is not None:
        amplitudes = emitter_amplitudes.to(device=targets.device, dtype=targets.dtype)
        response = response * amplitudes[None, None, :]
    return response.squeeze(0) if unbatched else response


class FourierFeatures(nn.Module):
    """Sinusoidal coordinate features for high-frequency acoustic fields."""

    def __init__(
        self,
        input_dim: int = 3,
        num_bands: int = 8,
        min_frequency: float = 1.0,
        max_frequency: float = 128.0,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)
        frequencies = torch.logspace(
            torch.log10(torch.tensor(float(min_frequency))),
            torch.log10(torch.tensor(float(max_frequency))),
            steps=self.num_bands,
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.num_bands

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        pieces = [x] if self.include_input else []
        phases = x[..., None, :] * self.frequencies.to(x)[..., None] * (2.0 * torch.pi)
        pieces.extend([torch.sin(phases).flatten(-2), torch.cos(phases).flatten(-2)])
        return torch.cat(pieces, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


def _make_mlp(input_dim: int, width: int, output_dim: int, blocks: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.SiLU()]
    layers.extend(ResidualBlock(width, dropout) for _ in range(blocks))
    layers.extend([nn.LayerNorm(width), nn.Linear(width, output_dim)])
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class AdditiveResidualSurrogateConfig:
    geometry_dim: int = 61
    geometry_latent_dim: int = 256
    target_latent_dim: int = 128
    hidden_dim: int = 512
    num_emitters: int = 256
    target_fourier_bands: int = 8
    target_category_count: int = 3
    target_category_dim: int = 0
    geometry_blocks: int = 4
    target_blocks: int = 3
    decoder_blocks: int = 5
    dropout: float = 0.0
    use_free_field_features: bool = True


class AdditiveResidualBEMSurrogate(nn.Module):
    """Predict the additive BEM correction for a hand-conditioned scene."""

    def __init__(
        self,
        emitter_centers_m: Tensor,
        wave_number: float,
        emitter_amplitudes: Tensor | None = None,
        config: AdditiveResidualSurrogateConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AdditiveResidualSurrogateConfig()
        if emitter_centers_m.shape != (self.config.num_emitters, 3):
            raise ValueError(
                f"emitter_centers_m must have shape ({self.config.num_emitters}, 3), "
                f"got {tuple(emitter_centers_m.shape)}"
            )
        self.register_buffer("emitter_centers_m", emitter_centers_m.detach().clone().float())
        amplitudes = torch.ones(self.config.num_emitters) if emitter_amplitudes is None else emitter_amplitudes
        if amplitudes.shape != (self.config.num_emitters,):
            raise ValueError(f"emitter_amplitudes must have shape ({self.config.num_emitters},)")
        self.register_buffer("emitter_amplitudes", amplitudes.detach().clone().float())
        self.wave_number = float(wave_number)

        self.geometry_encoder = _make_mlp(
            self.config.geometry_dim,
            self.config.geometry_latent_dim,
            self.config.geometry_latent_dim,
            self.config.geometry_blocks,
            self.config.dropout,
        )
        self.target_fourier = FourierFeatures(
            input_dim=3,
            num_bands=self.config.target_fourier_bands,
            min_frequency=1.0,
            max_frequency=128.0,
            include_input=True,
        )
        self.target_category = (
            nn.Embedding(self.config.target_category_count, self.config.target_category_dim)
            if self.config.target_category_dim > 0
            else None
        )
        target_input_dim = self.target_fourier.output_dim + max(0, self.config.target_category_dim)
        self.target_encoder = _make_mlp(
            target_input_dim,
            self.config.target_latent_dim,
            self.config.target_latent_dim,
            self.config.target_blocks,
            self.config.dropout,
        )
        free_dim = 2 * self.config.num_emitters if self.config.use_free_field_features else 0
        decoder_input_dim = self.config.geometry_latent_dim + self.config.target_latent_dim + free_dim
        self.residual_decoder = _make_mlp(
            decoder_input_dim,
            self.config.hidden_dim,
            2 * self.config.num_emitters,
            self.config.decoder_blocks,
            self.config.dropout,
        )

    def encode_geometry(
        self,
        shape_beta: Tensor,
        hand_pose: Tensor,
        root_rotation: Tensor,
        root_translation_m: Tensor,
    ) -> Tensor:
        geometry = torch.cat([shape_beta, hand_pose, root_rotation, root_translation_m], dim=-1)
        if geometry.shape[-1] != self.config.geometry_dim:
            raise ValueError(f"geometry feature dim must be {self.config.geometry_dim}, got {geometry.shape[-1]}")
        return self.geometry_encoder(geometry)

    def encode_targets(self, target_world_m: Tensor, target_category: Tensor | None = None) -> Tensor:
        targets, unbatched = _as_batched_targets(target_world_m)
        encoded_xyz = self.target_fourier(targets)
        if self.target_category is not None:
            if target_category is None:
                categories = torch.zeros(targets.shape[:2], dtype=torch.long, device=targets.device)
            else:
                categories = target_category.to(device=targets.device, dtype=torch.long)
                if categories.ndim == 1:
                    categories = categories.unsqueeze(0)
                if categories.shape != targets.shape[:2]:
                    raise ValueError(
                        f"target_category must have shape {targets.shape[:2]}, got {tuple(categories.shape)}"
                    )
            encoded_category = self.target_category(categories.clamp(0, self.config.target_category_count - 1))
            encoded_xyz = torch.cat([encoded_xyz, encoded_category], dim=-1)
        encoded = self.target_encoder(encoded_xyz)
        return encoded.squeeze(0) if unbatched else encoded

    def free_field_response(self, target_world_m: Tensor) -> Tensor:
        return compute_point_free_field_response(
            target_world_m,
            self.emitter_centers_m,
            self.wave_number,
            self.emitter_amplitudes,
        )

    def forward(
        self,
        shape_beta: Tensor,
        hand_pose: Tensor,
        root_rotation: Tensor,
        root_translation_m: Tensor,
        target_world_m: Tensor,
        target_category: Tensor | None = None,
    ) -> dict[str, Tensor]:
        targets, unbatched = _as_batched_targets(target_world_m)
        batch_size, target_count = targets.shape[:2]

        g = self.encode_geometry(shape_beta, hand_pose, root_rotation, root_translation_m)
        g = _expand_geometry_feature(g, batch_size)
        g = g[:, None, :].expand(batch_size, target_count, -1)
        t = self.encode_targets(targets, target_category)

        h_free = self.free_field_response(targets)
        decoder_inputs = [g, t]
        if self.config.use_free_field_features:
            decoder_inputs.append(torch.cat([h_free.real, h_free.imag], dim=-1).to(targets.dtype))
        decoder_in = torch.cat(decoder_inputs, dim=-1)
        residual_raw = self.residual_decoder(decoder_in)
        d_real, d_imag = residual_raw.split(self.config.num_emitters, dim=-1)
        delta = torch.complex(d_real, d_imag)
        response = h_free + delta
        phase = torch.remainder(-torch.angle(response), 2.0 * torch.pi)

        result = {
            "response": response,
            "response_real": response.real,
            "response_imag": response.imag,
            "delta": delta,
            "delta_real": d_real,
            "delta_imag": d_imag,
            "free_response": h_free,
            "phase": phase,
        }
        if unbatched:
            return {key: value.squeeze(0) for key, value in result.items()}
        return result


def additive_residual_surrogate_loss(
    prediction: dict[str, Tensor],
    response_real: Tensor,
    response_imag: Tensor,
    *,
    residual_weight: float = 1.0,
    response_weight: float = 1.0,
    log_amplitude_weight: float = 0.0,
    phase_weight: float = 0.0,
    complex_loss: str = "mse",
    eps: float = 1.0e-6,
) -> dict[str, Tensor]:
    """Compute training losses against strict BEM response labels."""
    target = torch.complex(response_real, response_imag)
    if target.shape != prediction["response"].shape:
        raise ValueError(f"target response shape {target.shape} != prediction {prediction['response'].shape}")
    delta_target = target - prediction["free_response"]

    if complex_loss == "mse":
        residual_loss = F.mse_loss(prediction["delta"].real, delta_target.real) + F.mse_loss(
            prediction["delta"].imag, delta_target.imag
        )
        response_loss = F.mse_loss(prediction["response"].real, target.real) + F.mse_loss(
            prediction["response"].imag, target.imag
        )
    elif complex_loss == "nmse":
        target_power = target.abs().square().mean(dim=(-2, -1)).clamp_min(eps)
        residual_error = (prediction["delta"] - delta_target).abs().square().mean(dim=(-2, -1))
        response_error = (prediction["response"] - target).abs().square().mean(dim=(-2, -1))
        residual_loss = (residual_error / target_power).mean()
        response_loss = (response_error / target_power).mean()
    else:
        raise ValueError(f"Unsupported complex_loss: {complex_loss}")
    total = residual_weight * residual_loss + response_weight * response_loss
    losses = {
        "total": total,
        "residual": residual_loss,
        "response": response_loss,
    }

    if log_amplitude_weight:
        log_amp_loss = F.mse_loss(
            torch.log(torch.abs(prediction["response"]) + eps),
            torch.log(torch.abs(target) + eps),
        )
        losses["log_amplitude"] = log_amp_loss
        losses["total"] = losses["total"] + log_amplitude_weight * log_amp_loss

    if phase_weight:
        phase_error = torch.angle(prediction["response"]) - torch.angle(target)
        amplitude = torch.abs(target)
        weight = amplitude / (amplitude.detach().mean().clamp_min(eps))
        weight = weight.clamp(max=10.0)
        phase_loss = torch.mean(weight * (1.0 - torch.cos(phase_error)))
        losses["phase"] = phase_loss
        losses["total"] = losses["total"] + phase_weight * phase_loss

    return losses

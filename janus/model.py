"""Predictive world-model: encoder + latent predictor, trained JEPA-style.

The model never predicts pixels. It predicts the *embedding* of a future frame,
supervised by an EMA copy of its own encoder with a stop-gradient. That gives a
free, infinite label stream off a live video feed -- every frame that arrives is
the answer to a question the model already asked.

No BatchNorm anywhere: the frame stream is violently non-iid, so batch statistics
would drift with the scene and leak the future into the target.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class Encoder(nn.Module):
    """Small strided conv tower: (B,3,R,R) -> (B,dim)."""

    def __init__(self, dim: int = 256, width: int = 32):
        super().__init__()
        w = width
        chans = [(3, w), (w, 2 * w), (2 * w, 4 * w), (4 * w, 8 * w)]
        blocks = []
        for cin, cout in chans:
            blocks += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), _norm(cout), nn.SiLU()]
        self.tower = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w, dim), nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tower(x)
        h = F.adaptive_avg_pool2d(h, 1)
        return self.head(h)


class Predictor(nn.Module):
    """(context embeddings [, action]) -> predicted embedding at t + horizon.

    The action is the agent's own commanded motion over the horizon. Without it
    the predictor has to treat the agent's decisions as unexplained variance,
    and surprise conflates "the world did something" with "I did something".
    """

    def __init__(self, dim: int = 256, context: int = 2, hidden: int = 1024, action_dim: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(dim * context + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, ctx: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = ctx.flatten(1)
        if self.action_dim:
            h = torch.cat([h, action], dim=1)
        return self.net(h)


class WorldModel(nn.Module):
    """Online encoder + predictor, plus the EMA target encoder that supervises it."""

    def __init__(self, dim: int = 256, width: int = 32, context: int = 2, hidden: int = 1024,
                 ema_decay: float = 0.996, action_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.context = context
        self.ema_decay = ema_decay
        self.action_dim = action_dim
        self.encoder = Encoder(dim, width)
        self.predictor = Predictor(dim, context, hidden, action_dim)
        self.target = Encoder(dim, width)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        d = self.ema_decay
        for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(d).add_(po.detach(), alpha=1.0 - d)
        for bt, bo in zip(self.target.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def encode_context(self, ctx_frames: torch.Tensor) -> torch.Tensor:
        """ctx_frames: (B, C, 3, R, R) -> (B, C, dim)."""
        b, c = ctx_frames.shape[:2]
        z = self.encoder(ctx_frames.flatten(0, 1))
        return z.view(b, c, -1)

    def predict(self, ctx_frames: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_context(ctx_frames)
        return self.predictor(z, action), z[:, -1]

    def predict_from_z(self, z_ctx: torch.Tensor,
                       action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z_ctx: (B, C, dim) already-encoded context -- lets the server encode
        each frame once and reuse it as context for the next tick."""
        return self.predictor(z_ctx, action)

    @torch.no_grad()
    def encode_target(self, future_frame: torch.Tensor) -> torch.Tensor:
        return self.target(future_frame)


def prediction_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine distance in [0, 2]. This is the surprise signal."""
    return 1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge + off-diagonal covariance penalty.

    Without these the cheapest way to make prediction error vanish is to emit a
    constant embedding -- surprise goes to zero and the run *looks* like it
    learned something. The hinge makes every dimension carry >= 1 unit of std;
    the covariance term stops dimensions from duplicating each other.
    """
    if z.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = zc.shape
    cov = (zc.T @ zc) / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / d
    return var_loss, cov_loss


@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-9) -> float:
    """exp(entropy of the normalized singular-value spectrum). Collapse detector.

    A healthy 256-d embedding sits in the tens; a collapsed one falls toward 1.
    """
    if z.shape[0] < 2:
        return 0.0
    zc = (z - z.mean(dim=0, keepdim=True)).float()
    try:
        s = torch.linalg.svdvals(zc)
    except Exception:
        return float("nan")
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()).item())


def float_state_keys(model: nn.Module):
    """Stable ordering of the floating-point state we ship over the weight bus."""
    return sorted(k for k, v in model.state_dict().items() if v.is_floating_point())


def state_numel(model: nn.Module) -> int:
    sd = model.state_dict()
    return sum(sd[k].numel() for k in float_state_keys(model))

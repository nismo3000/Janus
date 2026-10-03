"""Continual backprop (Dohare et al., Nature 2024) on the predictor's hidden units.

The E3 baseline Janus has to beat. CBP keeps a running *contribution utility* per
hidden unit -- |activation| x L1 norm of the unit's outgoing weights -- and, at a
fixed replacement rate, re-initialises the least useful mature units: incoming
weights fresh, outgoing weights zero (so the reset is invisible on the wire), and
the optimizer state for those rows cleared. It restores plasticity by recycling
dead capacity *inside a fixed-size net*; it does not grow, does not route, and
treats every unit as fungible. That is the contrast with grow/decay: CBP recycles
units blindly at a constant rate; Janus is meant to grow on surprise and decay on
disuse, with the decayed weights archived rather than destroyed.

Applies to `Predictor.net`: Linear(in,h) -> LN -> SiLU -> Linear(h,h) -> LN -> SiLU -> Linear(h,dim).
Hidden layer k has incoming Linear at index 3k and outgoing Linear at index 3k+3.
"""

from typing import List

import torch
import torch.nn as nn


class ContinualBackprop:
    def __init__(self, predictor: nn.Module, opt: torch.optim.Optimizer,
                 rate: float = 1e-4, maturity: int = 100, decay: float = 0.99, seed: int = 0):
        self.net = predictor.net
        self.opt = opt
        self.rate, self.maturity, self.decay = rate, maturity, decay
        self.layers: List[tuple] = []           # (in_linear, act_index, out_linear)
        self.util: List[torch.Tensor] = []
        self.age: List[torch.Tensor] = []
        self.acts: List[torch.Tensor] = []
        self.accum: List[float] = []            # fractional replacements owed per layer
        self.gen = torch.Generator().manual_seed(seed + 101)
        self.replaced = 0
        lin = [i for i, m in enumerate(self.net) if isinstance(m, nn.Linear)]
        for a, b in zip(lin[:-1], lin[1:]):
            lin_in, lin_out = self.net[a], self.net[b]
            h = lin_in.out_features
            dev = lin_in.weight.device
            self.layers.append((lin_in, a + 2, lin_out))      # a+2 = the SiLU after LN
            self.util.append(torch.zeros(h, device=dev))
            self.age.append(torch.zeros(h, device=dev))
            self.acts.append(None)
            self.accum.append(0.0)
            self.net[a + 2].register_forward_hook(self._hook(len(self.layers) - 1))

    def _hook(self, k: int):
        def fn(_m, _i, out):
            self.acts[k] = out.detach()
        return fn

    @torch.no_grad()
    def step(self) -> int:
        """Call after opt.step(). Returns number of units reset this step."""
        n_reset = 0
        for k, (lin_in, _, lin_out) in enumerate(self.layers):
            a = self.acts[k]
            if a is None:
                continue
            contrib = a.abs().mean(0) * lin_out.weight.abs().sum(0)      # (h,)
            self.util[k].mul_(self.decay).add_((1 - self.decay) * contrib)
            self.age[k] += 1
            h = self.util[k].numel()
            self.accum[k] += self.rate * h
            n = int(self.accum[k])
            if n <= 0:
                continue
            self.accum[k] -= n
            eligible = self.age[k] >= self.maturity
            if eligible.sum() < n:
                continue
            u = self.util[k].clone()
            u[~eligible] = float("inf")
            idx = torch.topk(u, n, largest=False).indices
            # fresh incoming rows, zero outgoing columns, cleared optimizer state
            fan_in = lin_in.in_features
            bound = 1.0 / fan_in ** 0.5
            new_w = (torch.rand((n, fan_in), generator=self.gen) * 2 - 1) * bound
            lin_in.weight[idx] = new_w.to(lin_in.weight.device)
            if lin_in.bias is not None:
                lin_in.bias[idx] = 0.0
            lin_out.weight[:, idx] = 0.0
            for p, sl in ((lin_in.weight, (idx, slice(None))), (lin_in.bias, (idx,)),
                          (lin_out.weight, (slice(None), idx))):
                st = self.opt.state.get(p)
                if st:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key][sl] = 0.0
            self.util[k][idx] = 0.0
            self.age[k][idx] = 0
            n_reset += n
        self.replaced += n_reset
        return n_reset

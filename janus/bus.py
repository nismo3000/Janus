"""Shared-memory plumbing between the inferencer and the learner.

Two objects cross the process boundary:

  FrameRing  -- the inferencer writes every frame it sees; the learner samples
                (context, future) pairs out of it. One writer, one reader.
  WeightBus  -- the learner publishes fresh weights; the inferencer hot-swaps
                them mid-stream. Double-buffered seqlock so the reader can never
                observe a half-written parameter vector.
  DisplayBus -- the inferencer publishes a few uint8 panels (reality, the decoded
                prediction, the decoded dream head) plus a stats vector; the viewer
                process reads them for the side-by-side page. Optional.

Everything lives in pinned CPU shared memory. The weight vector is a few tens of
MB and gets published every couple of seconds, so the copy cost is noise -- and
it keeps the two CUDA contexts fully independent.
"""

import multiprocessing as mp
from typing import List, Optional

import torch
import torch.nn as nn

from .model import float_state_keys


class FrameRing:
    """Lock-free single-writer ring of uint8 frames, indexed by a monotonic counter."""

    def __init__(self, capacity: int, res: int, action_dim: int = 0):
        self.capacity = capacity
        self.res = res
        self.action_dim = action_dim
        self.buf = torch.zeros((capacity, res, res, 3), dtype=torch.uint8).share_memory_()
        # The agent's command at each frame rides alongside the pixels, same index.
        self.act = torch.zeros((capacity, max(1, action_dim)), dtype=torch.float32).share_memory_()
        self.count = mp.Value("q", 0)   # total frames ever written (monotonic)

    def write(self, frame_u8: torch.Tensor, action: Optional[torch.Tensor] = None) -> int:
        with self.count.get_lock():
            idx = self.count.value
            self.count.value = idx + 1
        self.buf[idx % self.capacity] = frame_u8
        if action is not None and self.action_dim:
            self.act[idx % self.capacity] = action
        return idx

    def total(self) -> int:
        with self.count.get_lock():
            return self.count.value

    def valid_range(self) -> tuple:
        """[lo, hi) of global indices still resident in the ring."""
        hi = self.total()
        lo = max(0, hi - self.capacity)
        # Leave a margin so we never read the slot the writer is mid-write on.
        return lo + 2, max(lo + 2, hi - 1)

    def gather(self, global_indices: List[int]) -> torch.Tensor:
        slots = [i % self.capacity for i in global_indices]
        return self.buf[slots]

    def gather_actions(self, global_indices: List[int]) -> torch.Tensor:
        slots = [i % self.capacity for i in global_indices]
        return self.act[slots]


class FlatLayout:
    """Where each float state tensor lives inside the flattened weight vector.

    The same sorted-key order the bus publishes in, so a flat vector can be bound
    straight onto a model as views: after `bind`, the model's parameters *are*
    slices of `flat`, and swapping weights is a pointer swap rather than a copy.
    """

    def __init__(self, model: nn.Module):
        sd = model.state_dict()
        self.entries = []                       # (key, offset, numel, shape)
        off = 0
        for k in float_state_keys(model):
            t = sd[k]
            self.entries.append((k, off, t.numel(), tuple(t.shape)))
            off += t.numel()
        self.numel = off

    def span(self, prefix: str) -> tuple:
        """[start, end) of every key under `prefix` (keys are sorted, so contiguous)."""
        hits = [(o, o + n) for k, o, n, _ in self.entries if k.startswith(prefix)]
        if not hits:
            raise KeyError(prefix)
        start, end = hits[0][0], hits[-1][1]
        assert end - start == sum(b - a for a, b in hits), f"{prefix} not contiguous"
        return start, end

    def bind(self, model: nn.Module, flat: torch.Tensor, prefix: str = "") -> None:
        """Point every float parameter/buffer of `model` at its slice of `flat`.

        `prefix` lets a sub-module (e.g. `model.target`) bind against a flat vector
        that holds only that sub-module's span.
        """
        params = dict(model.named_parameters())
        bufs = dict(model.named_buffers())
        base = self.span(prefix)[0] if prefix else 0
        for k, off, n, shape in self.entries:
            if prefix and not k.startswith(prefix):
                continue
            name = k[len(prefix):] if prefix else k
            view = flat[off - base:off - base + n].view(shape)
            if name in params:
                params[name].data = view
            elif name in bufs:
                mod_name, _, buf_name = name.rpartition(".")
                model.get_submodule(mod_name)._buffers[buf_name] = view
            else:
                raise KeyError(name)


class WeightBus:
    """Double-buffered publish/subscribe for a flattened float state_dict."""

    def __init__(self, numel: int):
        self.numel = numel
        self.slots = torch.zeros((2, numel), dtype=torch.float32).share_memory_()
        self.version = mp.Value("q", 0)

    # -- writer side -------------------------------------------------------
    def publish(self, model: nn.Module) -> int:
        sd = model.state_dict()
        keys = float_state_keys(model)
        flat = torch.cat([sd[k].detach().reshape(-1).float().cpu() for k in keys])
        with self.version.get_lock():
            v = self.version.value
            slot = (v + 1) % 2
            self.slots[slot].copy_(flat)
            self.version.value = v + 1
            return v + 1

    # -- reader side -------------------------------------------------------
    def current_version(self) -> int:
        with self.version.get_lock():
            return self.version.value

    def pull_into(self, dst: torch.Tensor, since: int) -> Optional[int]:
        """Copy the latest published vector into `dst` (a CPU tensor, ideally pinned)
        if newer than `since`. The lock is held only for the memcpy."""
        with self.version.get_lock():
            v = self.version.value
            if v <= since:
                return None
            dst.copy_(self.slots[v % 2])
        return v

    def pull(self, model: nn.Module, since: int) -> Optional[int]:
        """Load the latest published weights into `model` if newer than `since`."""
        with self.version.get_lock():
            v = self.version.value
            if v <= since:
                return None
            flat = self.slots[v % 2].clone()
        keys = float_state_keys(model)
        sd = model.state_dict()
        off = 0
        with torch.no_grad():
            for k in keys:
                t = sd[k]
                n = t.numel()
                t.copy_(flat[off:off + n].view_as(t).to(t.dtype))
                off += n
        return v


class Reservoir:
    """Vitter reservoir sample over the whole session -- the anti-forgetting memory.

    A pure recency buffer would let the model overwrite everything it knew about
    a scene the moment the scene changes, and the surprise signal would then be
    measuring drift rather than novelty.
    """

    def __init__(self, size: int, clip_len: int, res: int, generator: torch.Generator,
                 action_dim: int = 0):
        self.size = size
        self.clips = torch.zeros((size, clip_len, res, res, 3), dtype=torch.uint8)
        self.acts = torch.zeros((size, max(1, action_dim)), dtype=torch.float32)
        self.n_filled = 0
        self.n_seen = 0
        self.g = generator

    def offer(self, clip_u8: torch.Tensor, action: Optional[torch.Tensor] = None) -> None:
        self.n_seen += 1
        if self.n_filled < self.size:
            j = self.n_filled
            self.n_filled += 1
        else:
            j = int(torch.randint(0, self.n_seen, (1,), generator=self.g).item())
            if j >= self.size:
                return
        self.clips[j] = clip_u8
        if action is not None:
            self.acts[j] = action

    def sample(self, k: int) -> Optional[tuple]:
        if self.n_filled == 0:
            return None
        k = min(k, self.n_filled)
        idx = torch.randint(0, self.n_filled, (k,), generator=self.g)
        return self.clips[idx], self.acts[idx]


class DisplayBus:
    """Inferencer -> viewer: display panels + stats, and one command word back.

    Writer copies under the lock and bumps `seq`; the reader copies under the same
    lock, so a panel is never observed half-written. Everything is tiny (three
    96x96 frames), so holding the lock for the memcpy costs microseconds.
    """

    N_STATS = 24
    PANELS = ("real", "pred", "dream")
    CMD_RESYNC = 1

    def __init__(self, res: int):
        self.res = res
        self.frames = torch.zeros((len(self.PANELS), res, res, 3), dtype=torch.uint8).share_memory_()
        self.stats = torch.zeros(self.N_STATS, dtype=torch.float32).share_memory_()
        self.seq = mp.Value("q", 0)
        self.cmd = mp.Value("i", 0)

    def publish(self, panels, stats) -> int:
        """panels: dict name -> (res,res,3) uint8 CPU tensor (missing panels keep their
        last image); stats: sequence of floats, at most N_STATS."""
        with self.seq.get_lock():
            for i, name in enumerate(self.PANELS):
                t = panels.get(name)
                if t is not None:
                    self.frames[i].copy_(t)
            n = min(len(stats), self.N_STATS)
            self.stats[:n] = torch.as_tensor(list(stats[:n]), dtype=torch.float32)
            self.seq.value += 1
            return self.seq.value

    def read(self) -> tuple:
        """-> (seq, frames (P,res,res,3) uint8 copy, stats list)."""
        with self.seq.get_lock():
            return self.seq.value, self.frames.clone(), self.stats.tolist()

    def current_seq(self) -> int:
        with self.seq.get_lock():
            return self.seq.value

    def request(self, flag: int) -> None:
        with self.cmd.get_lock():
            self.cmd.value |= flag

    def take_commands(self) -> int:
        with self.cmd.get_lock():
            c = self.cmd.value
            self.cmd.value = 0
            return c


class DeviceWeightBus:
    """Single-pool weight bus (E2): the slots live on the GPU both processes share.

    torch.multiprocessing (spawn) hands the child processes CUDA IPC handles to the
    same device memory, so the learner's publish is one device-side cat plus a D2D
    copy, and the inferencer's swap is a pointer rebind. Nothing crosses host memory
    and no fetcher thread exists. This is the design rule "double-buffer the weights
    in one memory pool" -- the same shape as unified memory on Jetson/Spark.

    Three slots, not two: the writer must never touch the slot the reader is bound to
    (version c) nor one the reader might still have kernels in flight on. The reader
    publishes `consumed` = the version it is bound to *after* its stream has drained;
    the writer only overwrites the slot of version v-2 (== slot of v+1) once
    consumed >= v-1. If the reader is behind, the publish is skipped, never raced.
    """

    SLOTS = 3

    def __init__(self, numel: int, device):
        self.numel = numel
        self.device = torch.device(device)
        self.slots = torch.zeros((self.SLOTS, numel), dtype=torch.float32, device=self.device)
        self.version = mp.Value("q", 0)
        self.consumed = mp.Value("q", 0)
        self.skipped = mp.Value("q", 0)            # publishes refused because the reader was behind

    def slot(self, v: int) -> torch.Tensor:
        return self.slots[v % self.SLOTS]

    # -- writer side -------------------------------------------------------
    def publish(self, model: nn.Module) -> Optional[int]:
        with self.version.get_lock():
            v = self.version.value
        with self.consumed.get_lock():
            c = self.consumed.value
        if v >= 2 and c < v - 1:
            with self.skipped.get_lock():
                self.skipped.value += 1
            return None
        sd = model.state_dict()
        flat = torch.cat([sd[k].detach().reshape(-1).float() for k in float_state_keys(model)])
        self.slot(v + 1).copy_(flat.to(self.device, non_blocking=False))
        torch.cuda.current_stream(self.device).synchronize()   # landed before it is advertised
        with self.version.get_lock():
            self.version.value = v + 1
        return v + 1

    # -- reader side -------------------------------------------------------
    def current_version(self) -> int:
        with self.version.get_lock():
            return self.version.value

    def mark_consumed(self, v: int) -> None:
        with self.consumed.get_lock():
            self.consumed.value = max(self.consumed.value, v)

    def pull(self, model: nn.Module, since: int) -> Optional[int]:
        """Copy (not bind) the latest slot into `model`'s own tensors; used by the learner,
        which trains its private copy, and once by the inferencer before it binds."""
        with self.version.get_lock():
            v = self.version.value
        if v <= since:
            return None
        flat = self.slot(v)
        sd = model.state_dict()
        off = 0
        with torch.no_grad():
            for k in float_state_keys(model):
                t = sd[k]
                n = t.numel()
                t.copy_(flat[off:off + n].view_as(t).to(t.dtype))
                off += n
        return v

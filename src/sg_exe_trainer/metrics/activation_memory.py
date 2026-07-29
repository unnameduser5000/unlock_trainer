"""Utilities for measuring autograd saved tensor memory.

The tracker intentionally separates CUDA non-leaf tensors from all saved
tensors. Non-leaf CUDA saved tensors are the closest practical signal for
activation stash, while leaf tensors are more likely to be parameters or
parameter-like values saved by backward formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass
class _SavedTensorInfo:
    nbytes: int
    storage_key: tuple[str, int]
    storage_nbytes: int
    is_cuda: bool
    is_leaf: bool
    category: str


@dataclass
class _StorageLiveInfo:
    refcount: int
    nbytes: int
    is_cuda: bool
    is_leaf: bool
    category: str


@dataclass
class SavedTensorTracker:
    hidden_size: int | None = None
    vocab_size: int | None = None
    total_saves: int = 0
    total_saved_bytes: int = 0
    total_saved_cuda_bytes: int = 0
    total_saved_cuda_nonleaf_bytes: int = 0
    total_saved_cuda_leaf_bytes: int = 0
    live_bytes: int = 0
    live_cuda_bytes: int = 0
    live_cuda_nonleaf_bytes: int = 0
    live_cuda_leaf_bytes: int = 0
    live_cuda_unique_bytes: int = 0
    live_cuda_nonleaf_unique_bytes: int = 0
    live_cuda_leaf_unique_bytes: int = 0
    peak_live_bytes: int = 0
    peak_live_cuda_bytes: int = 0
    peak_live_cuda_nonleaf_bytes: int = 0
    peak_live_cuda_leaf_bytes: int = 0
    peak_live_cuda_unique_bytes: int = 0
    peak_live_cuda_nonleaf_unique_bytes: int = 0
    peak_live_cuda_leaf_unique_bytes: int = 0
    live_cuda_nonleaf_unique_hidden_bytes: int = 0
    live_cuda_nonleaf_unique_vocab_bytes: int = 0
    live_cuda_nonleaf_unique_attention_bytes: int = 0
    live_cuda_nonleaf_unique_other_bytes: int = 0
    peak_live_cuda_nonleaf_unique_hidden_bytes: int = 0
    peak_live_cuda_nonleaf_unique_vocab_bytes: int = 0
    peak_live_cuda_nonleaf_unique_attention_bytes: int = 0
    peak_live_cuda_nonleaf_unique_other_bytes: int = 0
    _next_token: int = 0
    _live: dict[int, _SavedTensorInfo] = field(default_factory=dict)
    _live_storages: dict[tuple[str, int, bool], _StorageLiveInfo] = field(default_factory=dict)

    def configure(self, *, hidden_size: int | None = None, vocab_size: int | None = None) -> None:
        if hidden_size is not None:
            self.hidden_size = int(hidden_size)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)

    def reset(self) -> None:
        self.total_saves = 0
        self.total_saved_bytes = 0
        self.total_saved_cuda_bytes = 0
        self.total_saved_cuda_nonleaf_bytes = 0
        self.total_saved_cuda_leaf_bytes = 0
        self.live_bytes = 0
        self.live_cuda_bytes = 0
        self.live_cuda_nonleaf_bytes = 0
        self.live_cuda_leaf_bytes = 0
        self.live_cuda_unique_bytes = 0
        self.live_cuda_nonleaf_unique_bytes = 0
        self.live_cuda_leaf_unique_bytes = 0
        self.peak_live_bytes = 0
        self.peak_live_cuda_bytes = 0
        self.peak_live_cuda_nonleaf_bytes = 0
        self.peak_live_cuda_leaf_bytes = 0
        self.peak_live_cuda_unique_bytes = 0
        self.peak_live_cuda_nonleaf_unique_bytes = 0
        self.peak_live_cuda_leaf_unique_bytes = 0
        self.live_cuda_nonleaf_unique_hidden_bytes = 0
        self.live_cuda_nonleaf_unique_vocab_bytes = 0
        self.live_cuda_nonleaf_unique_attention_bytes = 0
        self.live_cuda_nonleaf_unique_other_bytes = 0
        self.peak_live_cuda_nonleaf_unique_hidden_bytes = 0
        self.peak_live_cuda_nonleaf_unique_vocab_bytes = 0
        self.peak_live_cuda_nonleaf_unique_attention_bytes = 0
        self.peak_live_cuda_nonleaf_unique_other_bytes = 0
        self._next_token = 0
        self._live.clear()
        self._live_storages.clear()

    def _storage_key_and_nbytes(self, tensor: torch.Tensor) -> tuple[tuple[str, int], int]:
        try:
            storage = tensor.untyped_storage()
            return (str(tensor.device), int(storage.data_ptr())), int(storage.nbytes())
        except Exception:
            return (str(tensor.device), int(tensor.data_ptr())), tensor_nbytes(tensor)

    def _category(self, tensor: torch.Tensor) -> str:
        if tensor.is_leaf:
            return "leaf"
        shape = tuple(int(dim) for dim in tensor.shape)
        if self.vocab_size is not None and shape and shape[-1] == self.vocab_size:
            return "vocab"
        if self.hidden_size is not None and shape and shape[-1] == self.hidden_size:
            return "hidden"
        if tensor.dim() >= 4:
            return "attention"
        return "other"

    def pack(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        token = self._next_token
        self._next_token += 1
        nbytes = tensor_nbytes(tensor)
        storage_key, storage_nbytes = self._storage_key_and_nbytes(tensor)
        info = _SavedTensorInfo(
            nbytes=nbytes,
            storage_key=storage_key,
            storage_nbytes=storage_nbytes,
            is_cuda=bool(tensor.is_cuda),
            is_leaf=bool(tensor.is_leaf),
            category=self._category(tensor),
        )
        self._live[token] = info

        self.total_saves += 1
        self.total_saved_bytes += nbytes
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)

        if info.is_cuda:
            self.total_saved_cuda_bytes += nbytes
            self.live_cuda_bytes += nbytes
            self.peak_live_cuda_bytes = max(self.peak_live_cuda_bytes, self.live_cuda_bytes)
            if info.is_leaf:
                self.total_saved_cuda_leaf_bytes += nbytes
                self.live_cuda_leaf_bytes += nbytes
                self.peak_live_cuda_leaf_bytes = max(
                    self.peak_live_cuda_leaf_bytes,
                    self.live_cuda_leaf_bytes,
                )
            else:
                self.total_saved_cuda_nonleaf_bytes += nbytes
                self.live_cuda_nonleaf_bytes += nbytes
                self.peak_live_cuda_nonleaf_bytes = max(
                    self.peak_live_cuda_nonleaf_bytes,
                    self.live_cuda_nonleaf_bytes,
                )
            self._add_live_storage(info)

        return tensor, token

    def _add_live_storage(self, info: _SavedTensorInfo) -> None:
        key = (*info.storage_key, info.is_leaf)
        live = self._live_storages.get(key)
        if live is not None:
            live.refcount += 1
            return
        live = _StorageLiveInfo(
            refcount=1,
            nbytes=info.storage_nbytes,
            is_cuda=info.is_cuda,
            is_leaf=info.is_leaf,
            category=info.category,
        )
        self._live_storages[key] = live
        if not info.is_cuda:
            return
        self.live_cuda_unique_bytes += info.storage_nbytes
        self.peak_live_cuda_unique_bytes = max(
            self.peak_live_cuda_unique_bytes,
            self.live_cuda_unique_bytes,
        )
        if info.is_leaf:
            self.live_cuda_leaf_unique_bytes += info.storage_nbytes
            self.peak_live_cuda_leaf_unique_bytes = max(
                self.peak_live_cuda_leaf_unique_bytes,
                self.live_cuda_leaf_unique_bytes,
            )
        else:
            self.live_cuda_nonleaf_unique_bytes += info.storage_nbytes
            self.peak_live_cuda_nonleaf_unique_bytes = max(
                self.peak_live_cuda_nonleaf_unique_bytes,
                self.live_cuda_nonleaf_unique_bytes,
            )
            self._add_category_bytes(info.category, info.storage_nbytes)

    def _add_category_bytes(self, category: str, nbytes: int) -> None:
        if category == "hidden":
            self.live_cuda_nonleaf_unique_hidden_bytes += nbytes
            self.peak_live_cuda_nonleaf_unique_hidden_bytes = max(
                self.peak_live_cuda_nonleaf_unique_hidden_bytes,
                self.live_cuda_nonleaf_unique_hidden_bytes,
            )
        elif category == "vocab":
            self.live_cuda_nonleaf_unique_vocab_bytes += nbytes
            self.peak_live_cuda_nonleaf_unique_vocab_bytes = max(
                self.peak_live_cuda_nonleaf_unique_vocab_bytes,
                self.live_cuda_nonleaf_unique_vocab_bytes,
            )
        elif category == "attention":
            self.live_cuda_nonleaf_unique_attention_bytes += nbytes
            self.peak_live_cuda_nonleaf_unique_attention_bytes = max(
                self.peak_live_cuda_nonleaf_unique_attention_bytes,
                self.live_cuda_nonleaf_unique_attention_bytes,
            )
        else:
            self.live_cuda_nonleaf_unique_other_bytes += nbytes
            self.peak_live_cuda_nonleaf_unique_other_bytes = max(
                self.peak_live_cuda_nonleaf_unique_other_bytes,
                self.live_cuda_nonleaf_unique_other_bytes,
            )

    def _remove_live_storage(self, info: _SavedTensorInfo) -> None:
        key = (*info.storage_key, info.is_leaf)
        live = self._live_storages.get(key)
        if live is None:
            return
        live.refcount -= 1
        if live.refcount > 0:
            return
        self._live_storages.pop(key, None)
        if not info.is_cuda:
            return
        self.live_cuda_unique_bytes = max(0, self.live_cuda_unique_bytes - info.storage_nbytes)
        if info.is_leaf:
            self.live_cuda_leaf_unique_bytes = max(
                0,
                self.live_cuda_leaf_unique_bytes - info.storage_nbytes,
            )
        else:
            self.live_cuda_nonleaf_unique_bytes = max(
                0,
                self.live_cuda_nonleaf_unique_bytes - info.storage_nbytes,
            )
            self._remove_category_bytes(live.category, info.storage_nbytes)

    def _remove_category_bytes(self, category: str, nbytes: int) -> None:
        if category == "hidden":
            self.live_cuda_nonleaf_unique_hidden_bytes = max(
                0,
                self.live_cuda_nonleaf_unique_hidden_bytes - nbytes,
            )
        elif category == "vocab":
            self.live_cuda_nonleaf_unique_vocab_bytes = max(
                0,
                self.live_cuda_nonleaf_unique_vocab_bytes - nbytes,
            )
        elif category == "attention":
            self.live_cuda_nonleaf_unique_attention_bytes = max(
                0,
                self.live_cuda_nonleaf_unique_attention_bytes - nbytes,
            )
        else:
            self.live_cuda_nonleaf_unique_other_bytes = max(
                0,
                self.live_cuda_nonleaf_unique_other_bytes - nbytes,
            )

    def unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, tuple) or len(packed) != 2:
            return packed
        tensor, token = packed
        info = self._live.pop(int(token), None)
        if info is not None:
            self.live_bytes = max(0, self.live_bytes - info.nbytes)
            if info.is_cuda:
                self.live_cuda_bytes = max(0, self.live_cuda_bytes - info.nbytes)
                if info.is_leaf:
                    self.live_cuda_leaf_bytes = max(0, self.live_cuda_leaf_bytes - info.nbytes)
                else:
                    self.live_cuda_nonleaf_bytes = max(0, self.live_cuda_nonleaf_bytes - info.nbytes)
                self._remove_live_storage(info)
        return tensor

    def snapshot(self) -> dict[str, int]:
        return {
            "autograd_saved_tensors": self.total_saves,
            "autograd_saved_bytes_total": self.total_saved_bytes,
            "autograd_saved_cuda_bytes_total": self.total_saved_cuda_bytes,
            "autograd_saved_cuda_nonleaf_bytes_total": self.total_saved_cuda_nonleaf_bytes,
            "autograd_saved_cuda_leaf_bytes_total": self.total_saved_cuda_leaf_bytes,
            "autograd_saved_bytes_peak": self.peak_live_bytes,
            "autograd_saved_cuda_bytes_peak": self.peak_live_cuda_bytes,
            "autograd_saved_cuda_nonleaf_bytes_peak": self.peak_live_cuda_nonleaf_bytes,
            "autograd_saved_cuda_leaf_bytes_peak": self.peak_live_cuda_leaf_bytes,
            "autograd_saved_cuda_unique_bytes_peak": self.peak_live_cuda_unique_bytes,
            "autograd_saved_cuda_nonleaf_unique_bytes_peak": self.peak_live_cuda_nonleaf_unique_bytes,
            "autograd_saved_cuda_leaf_unique_bytes_peak": self.peak_live_cuda_leaf_unique_bytes,
            "autograd_saved_cuda_nonleaf_unique_hidden_bytes_peak": self.peak_live_cuda_nonleaf_unique_hidden_bytes,
            "autograd_saved_cuda_nonleaf_unique_vocab_bytes_peak": self.peak_live_cuda_nonleaf_unique_vocab_bytes,
            "autograd_saved_cuda_nonleaf_unique_attention_bytes_peak": self.peak_live_cuda_nonleaf_unique_attention_bytes,
            "autograd_saved_cuda_nonleaf_unique_other_bytes_peak": self.peak_live_cuda_nonleaf_unique_other_bytes,
            "autograd_saved_cuda_bytes_live_final": self.live_cuda_bytes,
            "autograd_saved_cuda_nonleaf_bytes_live_final": self.live_cuda_nonleaf_bytes,
            "autograd_saved_cuda_unique_bytes_live_final": self.live_cuda_unique_bytes,
            "autograd_saved_cuda_nonleaf_unique_bytes_live_final": self.live_cuda_nonleaf_unique_bytes,
        }

# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for kvcached VMM operations."""

import torch

# Importing _C defines the PageAllocator / InternalPage classes. On the stable
# build it also registers the KV tensor ops as torch.ops.kvcached.*; on the
# classic build they are functions on _C. TORCH_TARGET_VERSION is the encoded
# target on a stable build, None on a classic build.
from kvcached import _C  # type: ignore[attr-defined]

if _C.TORCH_TARGET_VERSION is None:
    # KV tensor ops (classic pybind11 path).
    init_kvcached = _C.init_kvcached
    shutdown_kvcached = _C.shutdown_kvcached
    create_kv_tensors = _C.create_kv_tensors
    kv_tensors_created = _C.kv_tensors_created
    map_to_kv_tensors = _C.map_to_kv_tensors
    unmap_from_kv_tensors = _C.unmap_from_kv_tensors
else:
    # KV tensor ops (stable-ABI dispatcher path).
    init_kvcached = torch.ops.kvcached.init_kvcached.default
    shutdown_kvcached = torch.ops.kvcached.shutdown_kvcached.default
    create_kv_tensors = torch.ops.kvcached.create_kv_tensors.default
    kv_tensors_created = torch.ops.kvcached.kv_tensors_created.default
    map_to_kv_tensors = torch.ops.kvcached.map_to_kv_tensors.default
    unmap_from_kv_tensors = torch.ops.kvcached.unmap_from_kv_tensors.default

# Page-management classes (pybind11).
PageAllocator = _C.PageAllocator
InternalPage = _C.InternalPage

__all__ = [
    "InternalPage",
    "PageAllocator",
    "create_kv_tensors",
    "init_kvcached",
    "kv_tensors_created",
    "map_to_kv_tensors",
    "shutdown_kvcached",
    "unmap_from_kv_tensors",
]

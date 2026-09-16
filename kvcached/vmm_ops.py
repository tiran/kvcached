# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for kvcached VMM operations."""

import torch

# Importing _C registers the KV tensor ops (torch.ops.kvcached.*) and defines
# the PageAllocator / InternalPage classes.
from kvcached import _C  # type: ignore[attr-defined]

# KV tensor ops (stable ABI).
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

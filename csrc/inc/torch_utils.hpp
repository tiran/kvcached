// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

// Compatibility hub for the two tensor backends: stable ABI when
// TORCH_TARGET_VERSION is defined (Torch >= 2.13, set by setup.py), classic
// ATen/c10 otherwise. Downstream code refers only to the kv_* aliases.
// TODO(drop @ torch>=2.13): drop the classic branch and make the stable path
// unconditional (applies to every #ifdef TORCH_TARGET_VERSION in csrc).
#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#else
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#endif

namespace kvcached {

#ifdef TORCH_TARGET_VERSION
using kv_tensor_t = torch::stable::Tensor;
using kv_device_t = torch::stable::Device;
using kv_scalar_t = torch::headeronly::ScalarType;
inline constexpr auto kv_device_type_cpu = torch::headeronly::kCPU;
#else
using kv_tensor_t = at::Tensor;
using kv_device_t = c10::Device;
using kv_scalar_t = c10::ScalarType;
inline constexpr auto kv_device_type_cpu = c10::kCPU;
#endif

// Map a raw element size to a ScalarType. Only the width matters, so any type
// of the right size works; the enumerators are shared by both backends.
static inline kv_scalar_t torch_dtype_from_size(size_t dtype_size) {
  using ST = kv_scalar_t;
  switch (dtype_size) {
  case 1:
    return ST::Char;
  case 2:
    return ST::Short;
  case 4:
    return ST::Int;
  case 8:
    return ST::Long;
  default:
    throw std::runtime_error("Unsupported dtype size: " +
                             std::to_string(dtype_size));
  }
}

// Element size of a bare ScalarType, needed before any tensor exists (to size
// the mapping). The stable ABI has no free function for it, so compute
// directly.
static inline size_t element_size(kv_scalar_t dtype) {
  using ST = kv_scalar_t;
  switch (dtype) {
  case ST::Byte:
  case ST::Char:
  case ST::Bool:
    return 1;
  case ST::Short:
  case ST::Half:
  case ST::BFloat16:
    return 2;
  case ST::Int:
  case ST::Float:
    return 4;
  case ST::Long:
  case ST::Double:
    return 8;
  default:
    throw std::runtime_error("Unsupported dtype for element_size");
  }
}

} // namespace kvcached

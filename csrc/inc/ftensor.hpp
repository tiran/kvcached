// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "constants.hpp"
#include "page.hpp"
#include "torch_utils.hpp"

namespace kvcached {

/* NOTE: FTensorAllocator is thread-safe but FTensor is not. */
class KVCACHED_HIDDEN FTensor {
public:
  FTensor(const std::string &name, size_t size, kv_scalar_t dtype,
          kv_device_t dev, std::shared_ptr<Page> zero_page,
          size_t page_size = 0);
  ~FTensor();
  bool map(offset_t offset);
  bool unmap(offset_t offset);

  inline kv_tensor_t get_tensor() noexcept { return tensor_; }

private:
  bool map_(Page *page, offset_t offset, bool set_access = true);
  bool set_access_(generic_ptr_t addr, size_t size);
  bool init_with_zero_();

  std::string name_;
  generic_ptr_t vaddr_;
  size_t size_;
  size_t page_size_;
  kv_scalar_t dtype_;
  kv_device_t dev_;
  std::shared_ptr<Page> zero_page_;

  kv_tensor_t tensor_;
  std::unordered_map<page_id_t, std::unique_ptr<Page>> mapping_;
};

} // namespace kvcached

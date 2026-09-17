// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

// Fail if the target is newer than the libtorch headers (else the binary
// silently needs absent symbols).
// TODO(drop @ torch>=2.15): PyTorch ships this guard (pytorch/pytorch#193962).
#if TORCH_TARGET_VERSION > TORCH_ABI_VERSION
#error "TORCH_TARGET_VERSION is newer than the libtorch headers this build "       \
    "compiles against. Lower it to <= TORCH_ABI_VERSION "                       \
    "(torch/headeronly/version.h) or build against a newer PyTorch."
#endif
#else
// Classic path: the at::Tensor <-> Python type caster for the pybind ops below.
#include <torch/csrc/utils/pybind.h>
#endif

#include "allocator.hpp"
#include "constants.hpp"
#include "page_allocator.hpp"
#include "torch_utils.hpp"

namespace py = pybind11;

namespace kvcached {

// ---------------------------------------------------------------------------
// KV tensor ops -- the only bindings that touch the tensor type.
//
// Stable build: registered via STABLE_TORCH_LIBRARY, reached as
// torch.ops.kvcached.*. Classic build: pybind functions on the module, reached
// as kvcached._C.*. vmm_ops.py hides the difference.
//
// size_t values cross the boundary as int64_t. Only the classic path holds the
// GIL, so it is released around the blocking allocator work (issue #371).
// ---------------------------------------------------------------------------

#ifdef TORCH_TARGET_VERSION
#define KVCACHED_RELEASE_GIL() ((void)0)
#else
#define KVCACHED_RELEASE_GIL() py::gil_scoped_release release
#endif

void init_kvcached(std::string dev_str, int64_t page_size,
                   bool contiguous_layout) {
  KVCACHED_RELEASE_GIL();
  FTensorAllocator::init(dev_str, static_cast<size_t>(page_size),
                         contiguous_layout);
}

void shutdown_kvcached() {
  KVCACHED_RELEASE_GIL();
  FTensorAllocator::shutdown();
}

std::vector<kv_tensor_t>
create_kv_tensors(int64_t size, int64_t dtype_size, std::string dev_str,
                  int64_t num_layers, int64_t num_kv_buffers, int64_t group_id,
                  bool unified_pool) {
  KVCACHED_RELEASE_GIL();
  auto allocator = FTensorAllocator::global_allocator(group_id);
  auto dtype_ = torch_dtype_from_size(static_cast<size_t>(dtype_size));
  return allocator->create_kv_tensors(static_cast<size_t>(size), dtype_,
                                      dev_str, num_layers, num_kv_buffers,
                                      unified_pool);
}

bool kv_tensors_created(int64_t group_id) {
  KVCACHED_RELEASE_GIL();
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->kv_tensors_created();
}

bool map_to_kv_tensors(std::vector<int64_t> offsets, int64_t group_id) {
  KVCACHED_RELEASE_GIL();
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->map_to_kv_tensors(offsets);
}

bool unmap_from_kv_tensors(std::vector<int64_t> offsets, int64_t group_id) {
  KVCACHED_RELEASE_GIL();
  auto allocator = FTensorAllocator::global_allocator(group_id);
  return allocator->unmap_from_kv_tensors(offsets);
}

// ---------------------------------------------------------------------------
// PageAllocator / InternalPage bindings -- no torch types, so they stay on
// pybind11 regardless of the ABI path.
// ---------------------------------------------------------------------------

std::shared_ptr<PageAllocator> create_page_allocator(
    int64_t num_layers, int64_t mem_size_per_layer, int64_t page_size,
    int64_t world_size = 1, int64_t pp_rank = 0, bool async_sched = false,
    bool contiguous_layout = true, bool enable_page_prealloc = true,
    int64_t num_kv_buffers = 2, int64_t group_id = 0,
    const std::string &ipc_name = "") {

  return std::make_shared<PageAllocator>(
      num_layers, mem_size_per_layer, page_size, world_size, pp_rank,
      async_sched, contiguous_layout, enable_page_prealloc, num_kv_buffers,
      group_id, ipc_name);
}

// PageAllocator method bindings
void page_allocator_start_prealloc_thread(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->start_prealloc_thread();
}

void page_allocator_stop_prealloc_thread(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->stop_prealloc_thread();
}

std::shared_ptr<InternalPage>
page_allocator_alloc_page(std::shared_ptr<PageAllocator> allocator) {
  return allocator->alloc_page();
}

void page_allocator_free_page(std::shared_ptr<PageAllocator> allocator,
                              page_id_t page_id) {
  allocator->free_page(page_id);
}

void page_allocator_free_pages(std::shared_ptr<PageAllocator> allocator,
                               const std::vector<page_id_t> &page_ids) {
  allocator->free_pages(page_ids);
}

bool page_allocator_resize(std::shared_ptr<PageAllocator> allocator,
                           int64_t new_mem_size) {
  return allocator->resize(new_mem_size);
}

void page_allocator_trim(std::shared_ptr<PageAllocator> allocator) {
  allocator->trim();
}

void page_allocator_reset_free_page_order(
    std::shared_ptr<PageAllocator> allocator) {
  allocator->reset_free_page_order();
}

int64_t
page_allocator_get_num_free_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_free_pages();
}

int64_t
page_allocator_get_num_inuse_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_inuse_pages();
}

int64_t
page_allocator_get_num_total_pages(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_total_pages();
}

int64_t page_allocator_get_num_reserved_pages(
    std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_num_reserved_pages();
}

py::dict
page_allocator_get_page_state(std::shared_ptr<PageAllocator> allocator) {
  PageState state;
  {
    py::gil_scoped_release release;
    state = allocator->get_page_state();
  }
  py::dict result;
  result["total_pages"] = state.total_pages;
  result["free_pages"] = state.free_pages;
  result["inuse_pages"] = state.inuse_pages;
  result["reserved_pages"] = state.reserved_pages;
  return result;
}

int64_t page_allocator_get_avail_physical_pages(
    std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_avail_physical_pages();
}

int64_t page_allocator_check_and_get_resize_target(
    std::shared_ptr<PageAllocator> allocator, int64_t current_mem_size) {
  return allocator->check_and_get_resize_target(current_mem_size);
}

int64_t
page_allocator_get_resize_target(std::shared_ptr<PageAllocator> allocator) {
  return allocator->get_resize_target();
}

void page_allocator_set_broadcast_map_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastMapCallback callback) {
  allocator->set_broadcast_map_callback(callback);
}

void page_allocator_set_broadcast_unmap_callback(
    std::shared_ptr<PageAllocator> allocator, BroadcastUnmapCallback callback) {
  allocator->set_broadcast_unmap_callback(callback);
}

void page_allocator_set_use_worker_ipc(std::shared_ptr<PageAllocator> allocator,
                                       bool use_worker_ipc) {
  allocator->set_use_worker_ipc(use_worker_ipc);
}

page_id_t page_allocator_get_page_id(std::shared_ptr<PageAllocator> allocator,
                                     int64_t block_id, int64_t block_mem_size) {
  return allocator->get_page_id(block_id, block_mem_size);
}

// New function for grouping indices by page
std::unordered_map<page_id_t, std::vector<int64_t>>
page_allocator_group_indices_by_page(std::shared_ptr<PageAllocator> allocator,
                                     const std::vector<int64_t> &indices,
                                     int64_t block_mem_size) {
  return allocator->group_indices_by_page(indices, block_mem_size);
}

} // namespace kvcached

#ifdef TORCH_TARGET_VERSION
// Register the KV tensor ops in the "kvcached" dispatcher namespace.
STABLE_TORCH_LIBRARY(kvcached, m) {
  m.def("init_kvcached(str dev_str, int page_size=0, bool "
        "contiguous_layout=True) -> ()");
  m.def("shutdown_kvcached() -> ()");
  m.def("create_kv_tensors(int size, int dtype_size, str dev_str, int "
        "num_layers, int num_kv_buffers=2, int group_id=0, bool "
        "unified_pool=False) -> Tensor[]");
  m.def("kv_tensors_created(int group_id=0) -> bool");
  m.def("map_to_kv_tensors(int[] offsets, int group_id=0) -> bool");
  m.def("unmap_from_kv_tensors(int[] offsets, int group_id=0) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(kvcached, CompositeExplicitAutograd, m) {
  m.impl("init_kvcached", TORCH_BOX(&kvcached::init_kvcached));
  m.impl("shutdown_kvcached", TORCH_BOX(&kvcached::shutdown_kvcached));
  m.impl("create_kv_tensors", TORCH_BOX(&kvcached::create_kv_tensors));
  m.impl("kv_tensors_created", TORCH_BOX(&kvcached::kv_tensors_created));
  m.impl("map_to_kv_tensors", TORCH_BOX(&kvcached::map_to_kv_tensors));
  m.impl("unmap_from_kv_tensors", TORCH_BOX(&kvcached::unmap_from_kv_tensors));
}
#endif // TORCH_TARGET_VERSION

// Always hosts the torch-free PageAllocator / InternalPage classes; the classic
// build also registers the KV tensor ops here (the stable build uses the
// dispatcher above).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kvcached VMM plugin";

  // Stable-ABI target, or None on a classic build; the ABI path _C uses (see
  // vmm_ops.py).
#ifdef TORCH_TARGET_VERSION
  m.attr("TORCH_TARGET_VERSION") =
      py::int_(static_cast<uint64_t>(TORCH_TARGET_VERSION));
#else
  m.attr("TORCH_TARGET_VERSION") = py::none();
#endif

#ifndef TORCH_TARGET_VERSION
  // KV tensor ops (classic path); the stable path uses the dispatcher above.
  m.def("init_kvcached", &kvcached::init_kvcached, "Initialize kvcached",
        py::arg("dev_str"), py::arg("page_size") = 0,
        py::arg("contiguous_layout") = true);
  m.def("shutdown_kvcached", &kvcached::shutdown_kvcached, "Shutdown kvcached");
  m.def("create_kv_tensors", &kvcached::create_kv_tensors, "create_kv_tensors",
        py::arg("size"), py::arg("dtype_size"), py::arg("dev_str"),
        py::arg("num_layers"), py::arg("num_kv_buffers") = 2,
        py::arg("group_id") = 0, py::arg("unified_pool") = false);
  m.def("kv_tensors_created", &kvcached::kv_tensors_created,
        "kv_tensors_created", py::arg("group_id") = 0);
  m.def("map_to_kv_tensors", &kvcached::map_to_kv_tensors, "map_to_kv_tensors",
        py::arg("offsets"), py::arg("group_id") = 0);
  m.def("unmap_from_kv_tensors", &kvcached::unmap_from_kv_tensors,
        "unmap_from_kv_tensors", py::arg("offsets"), py::arg("group_id") = 0);
#endif

  // PageAllocator bindings
  py::class_<kvcached::PageAllocator, std::shared_ptr<kvcached::PageAllocator>>(
      m, "PageAllocator")
      .def(py::init(&kvcached::create_page_allocator), py::arg("num_layers"),
           py::arg("mem_size_per_layer"), py::arg("page_size"),
           py::arg("world_size") = 1, py::arg("pp_rank") = 0,
           py::arg("async_sched") = false, py::arg("contiguous_layout") = true,
           py::arg("enable_page_prealloc") = true,
           py::arg("num_kv_buffers") = 2, py::arg("group_id") = 0,
           py::arg("ipc_name") = "")
      .def("start_prealloc_thread",
           &kvcached::page_allocator_start_prealloc_thread)
      // The bindings below can block inside the allocator (alloc_page waits on
      // a condition variable for the prealloc worker; free/resize/trim unmap
      // pages; stop joins the worker). They must not hold the GIL while
      // blocked: the prealloc worker needs the GIL to run the Python broadcast
      // callback, and a caller parked in here with the GIL held deadlocks the
      // whole process (issue #371).
      .def("stop_prealloc_thread",
           &kvcached::page_allocator_stop_prealloc_thread,
           py::call_guard<py::gil_scoped_release>())
      .def("alloc_page", &kvcached::page_allocator_alloc_page,
           py::call_guard<py::gil_scoped_release>())
      .def("free_page", &kvcached::page_allocator_free_page,
           py::call_guard<py::gil_scoped_release>())
      .def("free_pages", &kvcached::page_allocator_free_pages,
           py::call_guard<py::gil_scoped_release>())
      .def("resize", &kvcached::page_allocator_resize,
           py::call_guard<py::gil_scoped_release>())
      .def("trim", &kvcached::page_allocator_trim,
           py::call_guard<py::gil_scoped_release>())
      .def("reset_free_page_order",
           &kvcached::page_allocator_reset_free_page_order)
      .def("get_num_free_pages", &kvcached::page_allocator_get_num_free_pages)
      .def("get_num_inuse_pages", &kvcached::page_allocator_get_num_inuse_pages)
      .def("get_num_total_pages", &kvcached::page_allocator_get_num_total_pages)
      .def("get_num_reserved_pages",
           &kvcached::page_allocator_get_num_reserved_pages)
      .def("get_page_state", &kvcached::page_allocator_get_page_state)
      .def("get_avail_physical_pages",
           &kvcached::page_allocator_get_avail_physical_pages)
      .def("check_and_get_resize_target",
           &kvcached::page_allocator_check_and_get_resize_target)
      .def("get_resize_target", &kvcached::page_allocator_get_resize_target)
      .def("get_page_id", &kvcached::page_allocator_get_page_id)
      .def("group_indices_by_page",
           &kvcached::page_allocator_group_indices_by_page)
      .def("set_broadcast_map_callback",
           &kvcached::page_allocator_set_broadcast_map_callback)
      .def("set_broadcast_unmap_callback",
           &kvcached::page_allocator_set_broadcast_unmap_callback)
      .def("set_use_worker_ipc", &kvcached::page_allocator_set_use_worker_ipc);

  // InternalPage bindings (now as independent class)
  py::class_<kvcached::InternalPage, std::shared_ptr<kvcached::InternalPage>>(
      m, "InternalPage")
      .def(py::init<kvcached::page_id_t, int64_t>(), py::arg("page_id"),
           py::arg("page_size"))
      .def_readonly("page_id", &kvcached::InternalPage::page_id)
      .def_readonly("page_size", &kvcached::InternalPage::page_size)
      .def("init", &kvcached::InternalPage::init)
      .def("alloc", &kvcached::InternalPage::alloc)
      .def("free", &kvcached::InternalPage::free)
      .def("free_batch", &kvcached::InternalPage::free_batch)
      .def("empty", &kvcached::InternalPage::empty)
      .def("full", &kvcached::InternalPage::full)
      .def("num_free_blocks", &kvcached::InternalPage::num_free_blocks)
      .def("get_free_blocks", &kvcached::InternalPage::get_free_blocks)
      .def_static("get_block_range", &kvcached::InternalPage::get_block_range)
      .def_static("get_num_blocks", &kvcached::InternalPage::get_num_blocks);
}

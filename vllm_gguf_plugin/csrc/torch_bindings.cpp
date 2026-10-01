#include <optional>

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

Tensor ggml_dequantize(Tensor W, int64_t type, int64_t m, int64_t n,
                       std::optional<ScalarType> dtype);
#ifndef VLLM_GGUF_LEGACY_ONLY
bool ggml_should_use_mmvq(int64_t type, int64_t cc, int64_t batch);
int64_t ggml_dense_upstream_capabilities(Tensor W, Tensor X, int64_t type,
                                         int64_t row);
Tensor ggml_dense_mmvf(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmvq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmf(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dequantize_upstream(Tensor W, int64_t type, int64_t m, int64_t n,
                                std::optional<ScalarType> dtype);
#endif
Tensor ggml_mul_mat_vec_a8_legacy(Tensor W, Tensor X, int64_t type,
                                  int64_t row);
Tensor ggml_mul_mat_a8_legacy(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids,
                   Tensor expert_ids, Tensor num_tokens_post_padded,
                   int64_t type, int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int64_t top_k,
                       int64_t type, int64_t row, int64_t tokens);
#ifndef VLLM_GGUF_LEGACY_ONLY
Tensor ggml_moe_a8_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                            int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                         int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_mmvq(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                     int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_mmq(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                    int64_t row, int64_t top_k, int64_t tokens,
                    std::optional<Tensor> expert_ids,
                    std::optional<Tensor> padded_count);
Tensor ggml_moe_grouped_dense(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                              int64_t row, int64_t top_k, int64_t tokens);
#endif
int64_t ggml_moe_get_block_size(int64_t type);

STABLE_TORCH_LIBRARY(_C_gguf, ops) {
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.def("ggml_should_use_mmvq(int type, int cc, int batch) -> bool");
  ops.def(
      "ggml_dense_upstream_capabilities(Tensor W, Tensor X, int type, SymInt "
      "row) -> int");
  ops.def(
      "ggml_dense_mmvf(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_dense_mmvq(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("ggml_dense_mmq(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("ggml_dense_mmf(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_dense_blas(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
#endif
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.def(
      "ggml_dequantize_upstream(Tensor W, int type, SymInt m, SymInt n, "
      "ScalarType? dtype) -> Tensor");
#endif
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.def(
      "ggml_moe_a8_upstream(Tensor X, Tensor W, Tensor topk_ids, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_upstream(Tensor X, Tensor W, Tensor topk_ids, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_mmvq(Tensor X, Tensor W, Tensor topk_ids, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  // Optional metadata makes topk_ids a flat sorted route list (tile size 16).
  // Omitting both tensors preserves the original raw-ID contract.
  ops.def(
      "ggml_moe_mmq(Tensor X, Tensor W, Tensor topk_ids, "
      "int type, SymInt row, SymInt top_k, SymInt tokens, "
      "Tensor? expert_ids=None, Tensor? padded_count=None) -> Tensor");
  ops.def(
      "ggml_moe_grouped_dense(Tensor X, Tensor W, Tensor topk_ids, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
#endif
  ops.def("ggml_moe_get_block_size(int type) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("ggml_dequantize", TORCH_BOX(&ggml_dequantize));
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.impl("ggml_dequantize_upstream", TORCH_BOX(&ggml_dequantize_upstream));
  ops.impl("ggml_dense_mmvf", TORCH_BOX(&ggml_dense_mmvf));
  ops.impl("ggml_dense_mmvq", TORCH_BOX(&ggml_dense_mmvq));
  ops.impl("ggml_dense_mmq", TORCH_BOX(&ggml_dense_mmq));
  ops.impl("ggml_dense_mmf", TORCH_BOX(&ggml_dense_mmf));
  ops.impl("ggml_dense_blas", TORCH_BOX(&ggml_dense_blas));
#endif
  // Keep the historical op names as fixed legacy entry points. Python selects
  // explicit upstream ops before crossing the Torch/C++ boundary.
  ops.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8_legacy));
  ops.impl("ggml_mul_mat_a8", TORCH_BOX(&ggml_mul_mat_a8_legacy));
  ops.impl("ggml_moe_a8", TORCH_BOX(&ggml_moe_a8));
  ops.impl("ggml_moe_a8_vec", TORCH_BOX(&ggml_moe_a8_vec));
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.impl("ggml_moe_a8_upstream", TORCH_BOX(&ggml_moe_a8_upstream));
  ops.impl("ggml_moe_upstream", TORCH_BOX(&ggml_moe_upstream));
  ops.impl("ggml_moe_mmvq", TORCH_BOX(&ggml_moe_mmvq));
  ops.impl("ggml_moe_mmq", TORCH_BOX(&ggml_moe_mmq));
  ops.impl("ggml_moe_grouped_dense", TORCH_BOX(&ggml_moe_grouped_dense));
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CompositeExplicitAutograd, ops) {
#ifndef VLLM_GGUF_LEGACY_ONLY
  ops.impl("ggml_should_use_mmvq", TORCH_BOX(&ggml_should_use_mmvq));
  ops.impl("ggml_dense_upstream_capabilities",
           TORCH_BOX(&ggml_dense_upstream_capabilities));
#endif
  ops.impl("ggml_moe_get_block_size", TORCH_BOX(&ggml_moe_get_block_size));
}

static struct PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT, "_C_gguf", nullptr, -1, nullptr,
};

extern "C" __attribute__((visibility("default"))) PyObject* PyInit__C_gguf(
    void) {
  return PyModule_Create(&_module_def);
}

# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## Kernel backend selection

On CUDA builds the plugin ships three kernel implementations for GGUF
quantized operations — upstream (llama.cpp kernels, enabled by default),
legacy (the plugin's original CUDA kernels), and Triton. The implementation
is chosen per operation through environment variables:

| Variable | Scope | Default |
| --- | --- | --- |
| `VLLM_GGUF_CUDA_KERNEL` | Global fallback for all operations | `auto` |
| `VLLM_GGUF_CUDA_DENSE_KERNEL` | Dense MMVQ / MMQ | unset (falls back to the global variable) |
| `VLLM_GGUF_CUDA_MOE_KERNEL` | Routed-expert MoE | unset (falls back to the global variable) |
| `VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL` | Dequantization | unset (falls back to the global variable) |

Each variable accepts `auto`, `upstream`, `legacy`, or `triton`. A
specialized variable overrides the global one; if neither is set, the
operation uses `auto`.

In `auto` mode the dispatcher prefers the upstream kernels, then falls back
to legacy CUDA, then to Triton, using the first backend that supports the
quantization type. Explicit modes are strict: when the selected
implementation cannot handle the quantization type, the call fails instead
of silently falling back. Dense dispatch uses llama.cpp's architecture- and
quantization-specific MMVQ thresholds: up to 8 rows by default, with earlier
transitions to MMQ for selected K-quants on architectures such as Ada and
Blackwell. ROCm builds always use the legacy kernels.

Example — pin dense kernels to legacy while keeping the automatic selection
elsewhere:

```bash
VLLM_GGUF_CUDA_DENSE_KERNEL=legacy vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 \
  --tokenizer Qwen/Qwen3-0.6B
```

See `doc/upstream.md` for details on the upstream integration, weight
storage padding, and the supported quantization types per backend.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.

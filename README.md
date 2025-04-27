# tiny-llm - LLM Serving in a Week

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

Still WIP and in very early stage. A tutorial on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

The goal is to learn the techniques behind efficiently serving an LLM model (i.e., Qwen2 models).

## Book

The tiny-llm book is available at [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/). You can follow the guide and start building.

## Community

You may join skyzh's Discord server and study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)

## Roadmap

| Week + Chapter | Topic                                                       | Code | Test | Doc |
| -------------- | ----------------------------------------------------------- | ---- | ---- | --- |
| 1.1            | Attention                                                   | ✅    | ✅   | ✅  |
| 1.2            | RoPE                                                        | ✅    | ✅   | ✅  |
| 1.3            | Grouped Query Attention                                     | ✅    | 🚧   | 🚧  |
| 1.4            | RMSNorm and MLP                                             | ✅    | 🚧   | 🚧  |
| 1.5            | Transformer Block                                           | ✅    | 🚧   | 🚧  |
| 1.6            | Load the Model                                              | ✅    | 🚧   | 🚧  |
| 1.7            | Generate Responses                                          | ✅    | ✅   | 🚧  |
| 2.1            | KV Cache                                                    | ✅    | 🚧   | 🚧  |
| 2.2            | Quantized Matmul and Linear (CPU)                           | 🚧    | 🚧   | 🚧  |
| 2.3            | Quantized Matmul and Linear (Metal)                         | 🚧    | 🚧   | 🚧  |
| 2.4            | Attention Kernel                                            | 🚧    | 🚧   | 🚧  |
| 2.5            | Softmax Kernel                                              | 🚧    | 🚧   | 🚧  |
| 2.6            | Prompt Cache / Multiple Requests                            | 🚧    | 🚧   | 🚧  |
| 2.7            | Benchmarking                                                | 🚧    | 🚧   | 🚧  |
| 3.1            | API Server                                                  | 🚧    | 🚧   | 🚧  |
| 3.2            | Continuous Batching                                         | 🚧    | 🚧   | 🚧  |


<!--

### Day 2: RoPE Embedding

Note there are traditional and non-traditional ropes.

**References**

* https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
* https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
* https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
* https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RoPE.html
* https://arxiv.org/abs/2104.09864

### Day 3: Grouped Query Attention

The Qwen2 models use Grouped Query Attention (GQA). GQA allows different dimensions for query and key/value.

**References**

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
* PyTorch API (the case where enable_gqa=True) https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* torchtune.modules.MultiHeadAttention https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html
* https://arxiv.org/abs/2305.13245v1

### Day 4: RMSNorm and MLP

RMSNorm needs to be accumulated over float32

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
* SiLU https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
* RMSNorm (note that it needs to accumulate at float32)

### Day 5: Transformer Block

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

### Day 6: Load the Model

We will use mlx-lm's loader to load the model. We will _steal_ the loaded parameters from the mlx model and
plug it into our own operators.

### Day 7: Generate Responses

* Qwen layers implementation in mlx-lm https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

Run `python main.py` and it should give you a reasonable response.

On my M4 Pro Mac Mini, my implementation gives 17 tokens per sec on Metal, versus 50 tokens per sec from the mlx-lm
Qwen2 implementation. Sadly, it also takes 4x memory than using the mlx-lm components as it does not support computation
over quantized parameters.

## Week 2

Quantization, implement softmax/linear/silu kernels, implement attention kernels, key-value cache and compression, attention masks, prompt cache.

## Week 3

Continuous batching, OpenAPI HTTP endpoint, integrate with other services.


-->

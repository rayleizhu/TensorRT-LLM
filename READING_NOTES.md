
## Big Picture

* https://nvidia.github.io/TensorRT-LLM/gpt_attention.html


## Entrance

- [examples/gpt/run.py](examples/gpt/run.py)
    - Ln 275, decoder = tensorrt_llm.runtime.GenerationSession()
    - Ln 285, decoder.setup()
    - Ln 294, decoder.decode()
- [tensorrt_llm/runtime/generation.py](tensorrt_llm/runtime/generation.py)
    - Ln 281, GenerationSession()
    - Ln 769, GenerationSession.setup(): prepare buffers for collecting inference logits, kv cache, etc.
    - Ln 1783, GenerationSession.decode(): call decode_stream() or decode_regular() to generate the next token
        - Ln 1878, GenerationSession.decode_stream(): use yield instead of return during generation
            - Ln 1773, GenerationSession.handle_per_step()
        - Ln 1887, GenerationSession.decode_regular(): return immediately when stop criterion met
            - Ln 1659, GenerationSession.handle_per_step()
    - Ln 1382, GenerationSession.handle_per_step()
        - Ln 1424, Ln 1052-1053, Ln 927-928 create arguments for LLM inference?
        - Ln 1441-1449, _Runtime.run(), launch the graph
    - Why are there two cuda graph instances? Ln 1440-1446


## Model Inference Call
- [examples/gpt/build.py](examples/gpt/build.py)
    - Ln 508-522, Model.forward() seems to be compiled together with Model.prepare_inputs() as a graph
- [tensorrt_llm/models/gpt/model.py](tensorrt_llm/models/gpt/model.py)
    - Ln 431, Model.prepare_inputs(), prepare Tensor placeholders and other arguments for Model.forward()
    - Ln 459, call the common GenerationMixin.prepare_basic_inputs()
- [tensorrt_llm/models/generation_mixin.py](tensorrt_llm/models/generation_mixin.py)
    - Ln 373-388 provide some common args consumed by Model.forward()


## Implementation of PagedAttention
- https://nvidia.github.io/TensorRT-LLM/gpt_attention.html#kv-cache-s
- https://nvidia.github.io/TensorRT-LLM/gpt_attention.html#context-and-generation-phases
    - The context and generation phases use different kernels
    - https://github.com/NVIDIA/TensorRT-LLM/issues/457
    - https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/tensorrt_llm/runtime/generation.py#L1030-L1045
    - cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention
    - cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention
- [tensorrt_llm/runtime/generation.py](tensorrt_llm/runtime/generation.py)
    -  Ln 798 - 807, the cache shape of paged kv cache
- [cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h)
    - Ln 1521 - 1523, Ln 1544, the KV Cache mechanism is integrated to the attention kernel


## Implementation of Attention Kernels
- [decoderMaskedMultiheadAttentionTemplate.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h)
    - This kernel seems to support the parallelization style like [FlashDecoding](https://princeton-nlp.github.io/flash-decoding/), see Ln 30.
- [Support for Flash-Decoding:speed up long-context inference · Issue #209 · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/issues/209)
- [Tactic running out of memory during Code Llama 34B build · Issue #29 · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/issues/29)
- https://nvidia.github.io/TensorRT-LLM/gpt_attention.html#cross-attention
- What is the cross attention mode of the decoderMaaskedMultihedAttention?
    - [cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h)
        - Ln 1150, Ln 1164-1166, Ln 1369-1405, it seems that when cross attention is enabled the KVcache won't be updated
        
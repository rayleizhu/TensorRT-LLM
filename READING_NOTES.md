
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




## Implementation of Attention Kernels

- [decoderMaskedMultiheadAttentionTemplate.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h)
    - This kernel seems to support the parallelization style like [FlashDecoding](https://princeton-nlp.github.io/flash-decoding/), see Ln 30.
- [Support for Flash-Decoding:speed up long-context inference · Issue #209 · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/issues/209)
- [Tactic running out of memory during Code Llama 34B build · Issue #29 · NVIDIA/TensorRT-LLM (github.com)](https://github.com/NVIDIA/TensorRT-LLM/issues/29)

## Implementation of PagedAttention

- [tensorrt_llm/runtime/generation.py, Ln 367, Ln 798](tensorrt_llm/runtime/generation.py)


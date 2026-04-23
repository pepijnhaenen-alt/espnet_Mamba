# Streaming ASR Recipes

This recipe set compares three causal or streaming-friendly ASR encoder choices under the same frontend, augmentation, normalization, and CTC-only training setup.

## Recipes

- [Streaming Mamba](conf/streaming_mamba.yaml): baseline streaming Mamba encoder.
- [Streaming S4](conf/streaming_s4.yaml): S4-based baseline built on the repo's existing state-space layers.
- [Streaming Transformer](conf/streaming_transformer.yaml): contextual block transformer baseline for a latency-oriented attention comparison.

## Common Setup

All three recipes use the same:

- 80-bin log-Mel frontend
- SpecAug configuration
- utterance-level mean/variance normalization
- CTC-only model configuration
- AdamW optimizer
- warmup learning-rate scheduler

## Comparison Goal

The intent is to compare encoder behavior under the same streaming-friendly recipe skeleton:

- S4 is the non-Mamba state-space baseline.
- Contextual block Transformer is the attention-based baseline.
- Streaming Mamba remains the main reference point.

For practical evaluation, use the same dataset, tokenization, and decode settings across the three recipes so that perplexity, accuracy, and latency differences reflect the encoder choice rather than training-side mismatches.
#include "tabular-candle-cascade-attn/kernels/candle/flashinfer_candle_op.h"

std::unique_ptr<BatchDecodeWithPagedKVCacheCandleWrapper> new_batch_decode_with_paged_kv_cache_candle_wrapper(unsigned int layout) {
  return std::unique_ptr<BatchDecodeWithPagedKVCacheCandleWrapper>(
    BatchPrefillWithDecodeKVCacheCandleWrapper::Create(layout)));
}

std::unique_ptr<BatchPrefillWithPagedKVCacheCandleWrapper> new_batch_prefill_with_paged_kv_cache_candle_wrapper(unsigned int layout) {
  return std::unique_ptr<BatchPrefillWithPagedKVCacheCandleWrapper>(
    BatchPrefillWithPagedKVCacheCandleWrapper::Create(layout)));
}

std::unique_ptr<BatchPrefillWithRaggedKVCacheCandleWrapper> new_batch_prefill_with_ragged_kv_cache_candle_wrapper(unsigned int layout) {
    return std::unique_ptr<BatchPrefillWithRaggedKVCacheCandleWrapper>(
        BatchPrefillWithRaggedKVCacheCandleWrapper::Create(layout)));
}

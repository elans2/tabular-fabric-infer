#pragma once
#include <memory>

typedef *const c_void CandleTensorPtr;

CandleTensorPtr single_decode_with_kv_cache(CandleTensorPtr q, CandleTensorPtr k, CandleTensorPtr v,
                                          CandleTensorPtr tmp, unsigned int pos_encoding_mode,
                                          unsigned int layout, float sm_scale, float rope_scale,
                                          float rope_theta);

std::vector<CandleTensorPtr> single_prefill_with_kv_cache(
    CandleTensorPtr q, CandleTensorPtr k, CandleTensorPtr v, CandleTensorPtr tmp, bool causal,
    unsigned int layout, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
    float sm_scale, float rope_scale, float rope_theta, bool return_lse);

void append_paged_kv_cache(CandleTensorPtr append_key, CandleTensorPtr append_value,
                           CandleTensorPtr append_indptr, CandleTensorPtr kv_data,
                           CandleTensorPtr kv_indices, CandleTensorPtr kv_indptr,
                           CandleTensorPtr kv_last_page_len, unsigned int layout);

std::vector<CandleTensorPtr> merge_state(CandleTensorPtr v_a, CandleTensorPtr s_a, CandleTensorPtr v_b,
                                       CandleTensorPtr s_b);

void merge_state_in_place(CandleTensorPtr v, CandleTensorPtr s, CandleTensorPtr v_other,
                          CandleTensorPtr s_other);

std::vector<CandleTensorPtr> merge_states(CandleTensorPtr v, CandleTensorPtr s);

std::vector<CandleTensorPtr> batch_decode_with_padded_kv_cache(
    CandleTensorPtr q, CandleTensorPtr k_padded, CandleTensorPtr v_padded, unsigned int layout,
    unsigned int pos_encoding_mode, float sm_scale, float rope_scale, float rope_theta,
    bool return_lse);

class BatchDecodeWithPagedKVCacheCandleWrapper {
 public:
  static BatchDecodeWithPagedKVCacheCandleWrapper Create(unsigned int layout) {
    return BatchDecodeWithPagedKVCacheCandleWrapper(layout);
  }
  void BeginForward(CandleTensorPtr workspace_buffer, CandleTensorPtr indptr,
                    CandleTensorPtr last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
                    unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
                    unsigned int pos_encoding_mode, CandleTensorPtr empty_data);
  void EndForward();
  std::vector<CandleTensorPtr> Forward(CandleTensorPtr q, CandleTensorPtr paged_kv_data,
                                     CandleTensorPtr paged_kv_indptr, CandleTensorPtr paged_kv_indices,
                                     CandleTensorPtr paged_kv_last_page_len,
                                     unsigned int pos_encoding_mode, float sm_scale,
                                     float rope_scale, float rope_theta, bool return_lse);

 private:
  BatchDecodeWithPagedKVCacheCandleWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};

std::unique_ptr<BatchDecodeWithPagedKVCacheCandleWrapper> new_batch_decode_with_paged_kv_cache_candle_wrapper(unsigned int layout);

class BatchPrefillWithPagedKVCacheCandleWrapper {
 public:
  static BatchPrefillWithPagedKVCacheCandleWrapper Create(unsigned int layout) {
    return BatchPrefillWithPagedKVCacheCandleWrapper(layout);
  }
  void BeginForward(CandleTensorPtr workspace_buffer, CandleTensorPtr qo_indptr,
                    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
                    unsigned int head_dim);
  void EndForward();
  std::vector<CandleTensorPtr> Forward(CandleTensorPtr q, CandleTensorPtr qo_indptr,
                                     CandleTensorPtr paged_kv_data, CandleTensorPtr paged_kv_indptr,
                                     CandleTensorPtr paged_kv_indices,
                                     CandleTensorPtr paged_kv_last_page_len, bool causal,
                                     unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                     float sm_scale, float rope_scale, float rope_theta,
                                     bool return_lse);

 private:
  BatchPrefillWithPagedKVCacheCandleWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchPrefillHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};


std::unique_ptr<BatchPrefillWithPagedKVCacheCandleWrapper> new_batch_prefill_with_paged_kv_cache_candle_wrapper(unsigned int layout);

class BatchPrefillWithRaggedKVCacheCandleWrapper {
 public:
  static BatchPrefillWithRaggedKVCacheCandleWrapper Create(unsigned int layout) {
    return BatchPrefillWithRaggedKVCacheCandleWrapper(layout);
  }
  void BeginForward(CandleTensorPtr workspace_buffer, CandleTensorPtr qo_indptr,
                    unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
                    unsigned int head_dim);
  void EndForward();
  std::vector<CandleTensorPtr> Forward(CandleTensorPtr q, CandleTensorPtr qo_indptr, CandleTensorPtr k,
                                     CandleTensorPtr v, CandleTensorPtr kv_indptr, bool causal,
                                     unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                     float sm_scale, float rope_scale, float rope_theta,
                                     bool return_lse);

 private:
  BatchPrefillWithRaggedKVCacheCandleWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchPrefillHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};

std::unique_ptr<BatchPrefillWithRaggedKVCacheCandleWrapper> new_batch_prefill_with_ragged_kv_cache_candle_wrapper(unsigned int layout);

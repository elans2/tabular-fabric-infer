#include "tabular-candle-cascade-attn/kernels/candle/flashinfer_candle_op.h"
#include "tabular-candle-cascade-attn/kernels/flashinfer/csrc/flashinfer_ops.h"

#include <iostream>

namespace flashinfer_candle_op {

    using torch::rand;

    std::unique_ptr <Tensor> new_tensor() {
        return std::make_unique<Tensor>();
    }

    UniqueTensor candle_single_decode_with_kv_cache(UniqueTensor q, UniqueTensor k, UniqueTensor v,
                                                    UniqueTensor tmp, unsigned int pos_encoding_mode,
                                                    unsigned int layout, float sm_scale, float rope_scale,
                                                    float rope_theta) {
        return single_decode_with_kv_cache(*q.get(), *k.get(), *v.get(), *tmp.get(), pos_encoding_mode, layout, sm_scale, rope_scale, rope_theta);
    }

    std::vector <UniqueTensor> candle_single_prefill_with_kv_cache(
            UniqueTensor q, UniqueTensor k, UniqueTensor v, UniqueTensor tmp, bool causal,
            unsigned int layout, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
            float sm_scale, float rope_scale, float rope_theta, bool return_lse) {
        return single_prefill_with_kv_cache(*q.get(), *k.get(), *v.get(), *tmp.get(), causal, layout, pos_encoding_mode,
                                            allow_fp16_qk_reduction, sm_scale, rope_scale, rope_theta,
                                            return_lse);
    }

    void candle_append_paged_kv_cache(UniqueTensor append_key, UniqueTensor append_value,
                                      UniqueTensor append_indptr, UniqueTensor kv_data,
                                      UniqueTensor kv_indices, UniqueTensor kv_indptr,
                                      UniqueTensor kv_last_page_len, unsigned int layout) {
        append_paged_kv_cache(*append_key.get(), *append_value.get(), *append_indptr.get(), *kv_data.get(), *kv_indices.get(), *kv_indptr.get(), *kv_last_page_len.get(),
                              layout);
    }

    std::vector <UniqueTensor> candle_merge_state(UniqueTensor v_a, UniqueTensor s_a, UniqueTensor v_b,
                                                  UniqueTensor s_b) {
        return merge_state(*v_a.get(), *s_a.get(), *v_b.get(), *s_b.get());
    }

    void candle_merge_state_in_place(UniqueTensor v, UniqueTensor s, UniqueTensor v_other,
                                     UniqueTensor s_other) {
        merge_state_in_place(*v.get(), *s.get(), *v_other.get(), *s_other.get());
    }

    std::vector <UniqueTensor> candle_merge_states(UniqueTensor v, UniqueTensor s) {
        return merge_states(*v.get(), *s.get());
    }

    std::vector <UniqueTensor> candle_batch_decode_with_padded_kv_cache(
            UniqueTensor q, UniqueTensor k_padded, UniqueTensor v_padded, unsigned int layout,
            unsigned int pos_encoding_mode, float sm_scale, float rope_scale, float rope_theta,
            bool return_lse) {
        return batch_decode_with_padded_kv_cache(*q.get(), *k_padded.get(), *v_padded.get(), layout, pos_encoding_mode,
                                                 sm_scale, rope_scale, rope_theta, return_lse);
    }

    void
    CandleBatchDecodeWithPagedKVCacheTorchWrapper::begin_forward(UniqueTensor workspace_buffer, UniqueTensor indptr,
                                                                 UniqueTensor last_page_len, unsigned int batch_size,
                                                                 unsigned int num_qo_heads,
                                                                 unsigned int num_kv_heads, unsigned int head_dim,
                                                                 unsigned int page_size,
                                                                 unsigned int pos_encoding_mode,
                                                                 UniqueTensor empty_data) {
        this.wrapper_.begin_forward(*workspace_buffer.get(), *indptr.get(), last_page_len, batch_size, num_qo_heads, num_kv_heads,
                                    head_dim, page_size, pos_encoding_mode, *empty_data.get());
    }

    std::vector <UniqueTensor>
    CandleBatchDecodeWithPagedKVCacheTorchWrapper::forward(UniqueTensor q, UniqueTensor paged_kv_data,
                                                           UniqueTensor paged_kv_indptr, UniqueTensor paged_kv_indices,
                                                           UniqueTensor paged_kv_last_page_len,
                                                           unsigned int pos_encoding_mode, float sm_scale,
                                                           float rope_scale, float rope_theta, bool return_lse) {
        return this.wrapper_.forward(*q.get(), *paged_kv_data.get(), *paged_kv_indptr.get(), *paged_kv_indices.get(), *paged_kv_last_page_len.get(),
                                     pos_encoding_mode, sm_scale, rope_scale, rope_theta, return_lse);
    }

    void CandleBatchDecodeWithPagedKVCacheTorchWrapper::end_forward() {
        this.wrapper_.end_forward();
    }

    std::unique_ptr <CandleBatchDecodeWithPagedKVCacheTorchWrapper>
    new_candle_batch_decode_with_paged_kv_cache_torch_wrapper(unsigned int layout) {
        return std::unique_ptr<CandleBatchDecodeWithPagedKVCacheTorchWrapper>(
                new CandleBatchDecodeWithPagedKVCacheTorchWrapper(layout));
    }

    std::unique_ptr <CandleBatchPrefillWithPagedKVCacheTorchWrapper>
    new_candle_batch_prefill_with_paged_kv_cache_torch_wrapper(unsigned int layout) {
        return std::unique_ptr<CandleBatchPrefillWithPagedKVCacheTorchWrapper>(
                new CandleBatchPrefillWithPagedKVCacheTorchWrapper(layout));
    }

    void
    CandleBatchPrefillWithPagedKVCacheTorchWrapper::begin_forward(UniqueTensor workspace_buffer, UniqueTensor qo_indptr,
                                                                  unsigned int batch_size, unsigned int num_qo_heads,
                                                                  unsigned int num_kv_heads,
                                                                  unsigned int head_dim) {

        this.wrapper_.BeginForward(*workspace_buffer.get(), *qo_indptr.get(), batch_size, num_qo_heads, num_kv_heads, head_dim);

    }

    std::vector <UniqueTensor>
    CandleBatchPrefillWithPagedKVCacheTorchWrapper::forward(UniqueTensor q, UniqueTensor qo_indptr,
                                                            UniqueTensor paged_kv_data, UniqueTensor paged_kv_indptr,
                                                            UniqueTensor paged_kv_indices,
                                                            UniqueTensor paged_kv_last_page_len, bool causal,
                                                            unsigned int pos_encoding_mode,
                                                            bool allow_fp16_qk_reduction,
                                                            float sm_scale, float rope_scale, float rope_theta,
                                                            bool return_lse) {
        return this.wrapper_.Forward(q, qo_indptr, paged_kv_data, paged_kv_indptr, paged_kv_indices,
                                     paged_kv_last_page_len, causal, pos_encoding_mode, allow_fp16_qk_reduction,
                                     sm_scale, rope_scale, rope_theta, return_lse);
    }

    void CandleBatchPrefillWithPagedKVCacheTorchWrapper::end_forward() {
        this.wrapper_.EndForward();
    }


}
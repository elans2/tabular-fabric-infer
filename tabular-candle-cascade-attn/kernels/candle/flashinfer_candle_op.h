#pragma once

#include <memory>

#include <torch/torch.h>

namespace flashinfer_candle_op {

    typedef std::unique_ptr <torch::Tensor> UniqueTensor;


    UniqueTensor new_tensor();


    UniqueTensor candle_single_decode_with_kv_cache(UniqueTensor q, UniqueTensor k, UniqueTensor v,
                                                    UniqueTensor tmp, unsigned int pos_encoding_mode,
                                                    unsigned int layout, float sm_scale, float rope_scale,
                                                    float rope_theta);

    std::vector <UniqueTensor> candle_single_prefill_with_kv_cache(
            UniqueTensor q, UniqueTensor k, UniqueTensor v, UniqueTensor tmp, bool causal,
            unsigned int layout, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
            float sm_scale, float rope_scale, float rope_theta, bool return_lse);

    void candle_append_paged_kv_cache(UniqueTensor append_key, UniqueTensor append_value,
                                      UniqueTensor append_indptr, UniqueTensor kv_data,
                                      UniqueTensor kv_indices, UniqueTensor kv_indptr,
                                      UniqueTensor kv_last_page_len, unsigned int layout);

    std::vector <UniqueTensor> candle_merge_state(UniqueTensor v_a, UniqueTensor s_a, UniqueTensor v_b,
                                                  UniqueTensor s_b);

    void candle_merge_state_in_place(UniqueTensor v, UniqueTensor s, UniqueTensor v_other,
                                     UniqueTensor s_other);

    std::vector <UniqueTensor> candle_merge_states(UniqueTensor v, UniqueTensor s);

    std::vector <UniqueTensor> candle_batch_decode_with_padded_kv_cache(
            UniqueTensor q, UniqueTensor k_padded, UniqueTensor v_padded, unsigned int layout,
            unsigned int pos_encoding_mode, float sm_scale, float rope_scale, float rope_theta,
            bool return_lse);

    class CandleBatchDecodeWithPagedKVCacheTorchWrapper {
    public:
        static CandleBatchDecodeWithPagedKVCacheTorchWrapper Create(unsigned int layout) {
            return CandleBatchDecodeWithPagedKVCacheTorchWrapper(layout);
        }

        void begin_forward(UniqueTensor workspace_buffer, UniqueTensor indptr,
                           UniqueTensor last_page_len, unsigned int batch_size, unsigned int num_qo_heads,
                           unsigned int num_kv_heads, unsigned int head_dim, unsigned int page_size,
                           unsigned int pos_encoding_mode, UniqueTensor empty_data);

        void end_forward();

        std::vector <UniqueTensor> forward(UniqueTensor q, UniqueTensor paged_kv_data,
                                           UniqueTensor paged_kv_indptr, UniqueTensor paged_kv_indices,
                                           UniqueTensor paged_kv_last_page_len,
                                           unsigned int pos_encoding_mode, float sm_scale,
                                           float rope_scale, float rope_theta, bool return_lse);

    private:
        CandleBatchDecodeWithPagedKVCacheTorchWrapper(unsigned int layout)
                : wrapper_(flashinfer::BatchDecodeWithPagedKVCachePyTorchWrapper(layout)) {}

        flashinfer::BatchDecodeWithPagedKVCachePyTorchWrapper wrapper_;
    };

    std::unique_ptr <CandleBatchDecodeWithPagedKVCacheTorchWrapper>
    new_candle_batch_decode_with_paged_kv_cache_torch_wrapper(unsigned int layout);

    class CandleBatchPrefillWithPagedKVCacheTorchWrapper {
    public:
        static CandleBatchPrefillWithPagedKVCacheTorchWrapper Create(unsigned int layout) {
            return CandleBatchPrefillWithPagedKVCacheTorchWrapper(layout);
        }

        void begin_forward(UniqueTensor workspace_buffer, UniqueTensor qo_indptr,
                           unsigned int batch_size, unsigned int num_qo_heads, unsigned int num_kv_heads,
                           unsigned int head_dim);

        void end_forward();

        std::vector <UniqueTensor> forward(UniqueTensor q, UniqueTensor qo_indptr,
                                           UniqueTensor paged_kv_data, UniqueTensor paged_kv_indptr,
                                           UniqueTensor paged_kv_indices,
                                           UniqueTensor paged_kv_last_page_len, bool causal,
                                           unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
                                           float sm_scale, float rope_scale, float rope_theta,
                                           bool return_lse);

    private:
        CandleBatchPrefillWithPagedKVCacheTorchWrapper(unsigned int layout)
                : wrapper_(flashinfer::BatchPrefillWithPagedKVCachePyTorchWrapper(layout)) {}

        flashinfer::BatchPrefillWithPagedKVCachePyTorchWrapper wrapper_;
    };


    std::unique_ptr <CandleBatchPrefillWithPagedKVCacheTorchWrapper>
    new_candle_batch_prefill_with_paged_kv_cache_torch_wrapper(unsigned int layout);


}
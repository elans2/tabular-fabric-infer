#[cxx::bridge(namespace = "flashinfer_candle_op")]
pub mod ffi {
    unsafe extern "C++" {
        include!("tabular-candle-cascade-attn/kernels/candle/flashinfer_candle_op.h");

        type Tensor;
        fn new_tensor() -> UniquePtr<Tensor>;

        type BatchPrefillWithPagedKVCacheTorchWrapper;

        fn new_batch_prefill_with_paged_kv_cache_torch_wrapper(
        ) -> UniquePtr<BatchPrefillWithPagedKVCacheTorchWrapper>;

        fn begin_forward(
            &self,
            workspace_buffer: UniqueTensor,
            indptr: UniqueTensor,
            last_page_len: UniqueTensor,
            batch_size: u32,
            num_qo_heads: u32,
            num_kv_heads: u32,
            head_dim: u32,
            page_size: u32,
            pos_encoding_mode: u32,
            empty_data: UniqueTensor,
        );

        fn end_forward(&self);

        fn forward(
            &self,
            q: UniqueTensor,
            paged_kv_data: UniqueTensor,
            paged_kv_indptr: UniqueTensor,
            paged_kv_indices: UniqueTensor,
            paged_kv_last_page_len: UniqueTensor,
            pos_encoding_mode: u32,
            sm_scale: f32,
            rope_scale: f32,
            rope_theta: f32,
            return_lse: bool,
        ) -> Vec<UniqueTensor>;

        type BatchDecodeWithPagedKVCacheTorchWrapper;

        fn new_batch_decode_with_paged_kv_cache_torch_wrapper(
        ) -> UniquePtr<BatchDecodeWithPagedKVCacheTorchWrapper>;

        fn begin_forward(
            &self,
            workspace_buffer: UniqueTensor,
            qo_indptr: UniqueTensor,
            batch_size: u32,
            num_qo_heads: u32,
            num_kv_heads: u32,
            head_dim: u32,
        );

        fn end_forward(&mut self);

        fn forward(
            &self,
            q: UniqueTensor,
            qo_indptr: UniqueTensor,
            paged_kv_data: UniqueTensor,
            paged_kv_indptr: UniqueTensor,
            paged_kv_indices: UniqueTensor,
            paged_kv_last_page_len: UniqueTensor,
            causal: bool,
            pos_encoding_mode: u32,
            allow_fp16_qk_reduction: bool,
            sm_scale: f32,
            rope_scale: f32,
            rope_theta: f32,
            return_lse: bool,
        ) -> Vec<UniqueTensor>;

        // fn single_decode_with_kv_cache(
        //     q: TorchTensorPtr,
        //     k: TorchTensorPtr,
        //     v: TorchTensorPtr,
        //     tmp: TorchTensorPtr,
        //     pos_encoding_mode: u32,
        //     layout: u32,
        //     sm_scale: f32,
        //     rope_scale: f32,
        //     rope_theta: f32,
        // ) -> TorchTensorPtr;
        //
        // fn single_prefill_with_kv_cache(
        //     q: TorchTensorPtr,
        //     k: TorchTensorPtr,
        //     v: TorchTensorPtr,
        //     tmp: TorchTensorPtr,
        //     causal: bool,
        //     layout: u32,
        //     pos_encoding_mode: u32,
        //     allow_fp16_qk_reduction: boo,
        //     sm_scale: f32,
        //     rope_scale: f32,
        //     rope_theta: f32,
        //     return_lse: bool,
        // ) -> TorchTensorPtr;
        //
        // fn append_paged_kv_cache(
        //     append_key: TorchTensorPtr,
        //     append_value: TorchTensorPtr,
        //     append_indptr: TorchTensorPtr,
        //     kv_data: TorchTensorPtr,
        //     kv_indices: TorchTensorPtr,
        //     kv_indptr: TorchTensorPtr,
        //     kv_last_page_len: TorchTensorPtr,
        //     layout: u32,
        // );
        //
        // fn merge_state(
        //     v_a: TorchTensorPtr,
        //     s_a: TorchTensorPtr,
        //     v_b: TorchTensorPtr,
        //     s_b: TorchTensorPtr,
        // ) -> Vec<TorchTensorPtr>;
        //
        // fn merge_state_in_place(
        //     v: TorchTensorPtr,
        //     s: TorchTensorPtr,
        //     v_other: TorchTensorPtr,
        //     s_other: TorchTensorPtr,
        // );
        //
        // fn merge_states(v: TorchTensorPtr, s: TorchTensorPtr) -> Vec<TorchTensorPtr>;
        //
        // fn batch_decode_with_padded_kv_cache(
        //     q: TorchTensorPtr,
        //     k_padded: TorchTensorPtr,
        //     v_padded: TorchTensorPtr,
        //     layout: u32,
        //     pos_encoding_mode: u32,
        //     sm_scale: f32,
        //     rope_scale: f32,
        //     rope_theta: f32,
        //     return_lse: bool,
        // ) -> Vec<TorchTensorPtr>;
    }
}
//
// fn main() {
//     let client = ffi::new_blobstore_client();
//     println!("Hello, world!");
// }
//
//
// use core::ffi::{c_int, c_void};
//
// extern "C" {
//     pub(crate) fn run_mha(
//         q_ptr: *const c_void,
//         k_ptr: *const c_void,
//         v_ptr: *const c_void,
//         o_ptr: *const c_void,
//         o_tmp_ptr: *const c_void,
//         softmax_lse_ptr: *const c_void,
//         cu_seqlens_q_ptr: *const i32,
//         cu_seqlens_k_ptr: *const i32,
//
//         q_row_stride: u32,
//         k_row_stride: u32,
//         v_row_stride: u32,
//         o_row_stride: u32,
//         o_tmp_row_stride: u32,
//
//         q_head_stride: u32,
//         k_head_stride: u32,
//         v_head_stride: u32,
//         o_head_stride: u32,
//         o_tmp_head_stride: u32,
//
//         b: u32,
//         h: u32,
//         d: u32,
//         softmax_scale: f32,
//
//         seqlen_q: u32,
//         seqlen_k: u32,
//
//         is_causal: c_int,
//         is_bf16: c_int,
//
//         multi_processor_count: i32,
//         num_splits: i32,
//     );
//
// }

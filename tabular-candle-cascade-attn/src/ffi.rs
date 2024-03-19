#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("tabular-candle-cascade-attn/kernels/candle/flashinfer_candle_op.h");

        // pub type TorchTensorPtr;
        //
        // pub type BatchPrefillWithPagedKVCacheTorchWrapper;
        // pub type BatchDecodeWithPagedKVCacheTorchWrapper;
        //
        // /*
        //     std::vector<TorchTensorPtr> batch_decode_with_padded_kv_cache(
        // TorchTensorPtr q, TorchTensorPtr k_padded, TorchTensorPtr v_padded, unsigned int layout,
        // unsigned int pos_encoding_mode, float sm_scale, float rope_scale, float rope_theta,
        // bool return_lse);
        //      */
        //
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

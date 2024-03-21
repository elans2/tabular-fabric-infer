use cxx::UniquePtr;

mod ffi;
//
// use candle;
// use candle::Module;
//

type UniqueTensor = UniquePtr<ffi::ffi::Tensor>;

pub fn new_tensor() -> UniquePtr<ffi::Tensor> {
    println!("new tensor ");
    ffi::new_tensor()
}

enum Layout {
    NHD = 0,
}

enum PosEncodingMode {
    None = 0,
    ROPE_LLAMA = 1,
}

pub struct BatchPrefillWithPagedKVCacheWrapper {
    wrapper: Box<UniquePtr<ffi::BatchPrefillWithPagedKVCacheTorchWrapper>>,
}

impl BatchPrefillWithPagedKVCacheWrapper {
    pub fn new(workspace_buffer: UniqueTensor, kv_layout: Layout) -> Self {
        let wrapper = Box::new(ffi::new_batch_prefill_with_paged_kv_cache_torch_wrapper());
        Self { wrapper }
    }

    pub fn begin_forward(
        &mut self,
        qo_indptr: &UniqueTensor,
        paged_kv_indptr: &UniqueTensor,
        paged_kv_indices: &UniqueTensor,
        paged_kv_last_page_len: &UniqueTensor,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        self.wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        );
    }

    pub fn forward(
        &mut self,
        q: &UniqueTensor,
        paged_kv_data: &UniqueTensor,
        causal: bool,
        pos_encoding_mode: PosEncodingMode,
        allow_fp16_qk_reduction: bool,
        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> UniqueTensor {
        self.wrapper.forward(
            q,
            paged_kv_data,
            causal,
            pos_encoding_mode as u32,
            allow_fp16_qk_reduction,
            sm_scale.unwrap_or(0.0),
            rope_scale.unwrap_or(0.0),
            rope_theta.unwrap_or(0.0),
        )
    }

    pub fn forward_return_lse(
        &mut self,
        q: &UniqueTensor,
        paged_kv_data: &UniqueTensor,
        causal: bool,
        pos_encoding_mode: PosEncodingMode,
        allow_fp16_qk_reduction: bool,
        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> (UniqueTensor, UniqueTensor) {
        self.wrapper.forward_return_lse(
            q,
            paged_kv_data,
            causal,
            pos_encoding_mode as u32,
            allow_fp16_qk_reduction,
            sm_scale.unwrap_or(0.0),
            rope_scale.unwrap_or(0.0),
            rope_theta.unwrap_or(0.0),
        )
    }

    pub fn end_forward(&mut self) {
        self.wrapper.end_forward();
    }

    pub fn reset_workspace_buffer(&mut self, new_workspace_buffer: &UniqueTensor) {
        self.wrapper.reset_workspace_buffer(new_workspace_buffer);
    }
}

pub struct BatchDecodeWithPagedKVCacheWrapper {
    wrapper: Box<UniquePtr<ffi::BatchDecodeWithPagedKVCacheTorchWrapper>>,
}

impl BatchDecodeWithPagedKVCacheWrapper {
    pub fn new(workspace_buffer: UniqueTensor, kv_layout: Layout) -> Self {
        let wrapper = Box::new(ffi::new_batch_decode_with_paged_kv_cache_torch_wrapper());
        Self { wrapper }
    }

    pub fn begin_forward(
        &self,
        indptr: &UniqueTensor,
        indices: &UniqueTensor,
        last_page_len: &UniqueTensor,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        pos_encoding_mode: PosEncodingMode,
        data_type: String,
    ) {
        self.wrapper.begin_forward(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode as u32,
            data_type,
        );
    }

    pub fn forward(
        &self,
        q: &UniqueTensor,
        paged_kv_data: &UniqueTensor,
        pos_encoding_mode: u32,
        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> UniqueTensor {
        self.wrapper.forward(
            q,
            paged_kv_data,
            pos_encoding_mode,
            sm_scale.unwrap_or(0.0),
            rope_scale.unwrap_or(0.0),
            rope_theta.unwrap_or(0.0),
        )
    }

    pub fn forward_return_lse(
        &self,
        q: &UniqueTensor,
        paged_kv_data: &UniqueTensor,
        pos_encoding_mode: u32,
        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> (UniqueTensor, UniqueTensor) {
        self.wrapper.forward_return_lse(
            q,
            paged_kv_data,
            pos_encoding_mode,
            sm_scale.unwrap_or(0.0),
            rope_scale.unwrap_or(0.0),
            rope_theta.unwrap_or(0.0),
        )
    }

    pub fn end_forward(&self) {
        self.wrapper.end_forward();
    }

    pub fn reset_workspace_buffer(&self, new_workspace_buffer: &UniqueTensor) {
        self.wrapper.reset_workspace_buffer(new_workspace_buffer);
    }
}

fn append_paged_kv_cache(
    append_key: &UniqueTensor,
    append_value: &UniqueTensor,
    append_indptr: &UniqueTensor,
    kv_data: &UniqueTensor,
    kv_indices: &UniqueTensor,
    kv_indptr: &UniqueTensor,
    kv_last_page_len: &UniqueTensor,
    layout: Layout,
) {
    ffi::append_paged_kv_cache(
        append_key,
        append_value,
        append_indptr,
        kv_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout,
    );
}

fn merge_state(
    v_a: &UniqueTensor,
    s_a: &UniqueTensor,
    v_b: &UniqueTensor,
    s_b: &UniqueTensor,
) -> Vec<UniqueTensor> {
    ffi::merge_state(v_a, s_a, v_b, s_b)
}

fn merge_state_in_place(
    v: &UniqueTensor,
    s: &UniqueTensor,
    v_other: &UniqueTensor,
    s_other: &UniqueTensor,
) {
    ffi::merge_state_in_place(v, s, v_other, s_other);
}

fn merge_states(v: &UniqueTensor, s: &UniqueTensor) -> Vec<UniqueTensor> {
    ffi::merge_states(v, s)
}

pub struct BatchPrefillWithSharedPrefixPagedKVCacheWrapper {}

impl BatchPrefillWithSharedPrefixPagedKVCacheWrapper {
    pub fn new(workspace_buffer: UniqueTensor, kv_layout: Layout) -> Self {
        todo!()
    }

    pub fn begin_forward(
        qo_indptr: &UniqueTensor,
        paged_kv_indptr: &UniqueTensor,
        paged_kv_indices: &UniqueTensor,
        paged_kv_last_page_len: &UniqueTensor,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
    }

    pub fn forward(
        q: &UniqueTensor,
        k_shared: &UniqueTensor,
        v_shared: &UniqueTensor,
        unique_kv_data: &UniqueTensor,
        causal: bool,
        allow_fp16_qk_reduction: bool,
        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> UniqueTensor {
        todo!()
    }

    pub fn end_forward() {}

    pub fn reset_workspace_buffer(new_workspace_buffer: &UniqueTensor) {
        todo!()
    }
}

pub struct BatchDecodeWithSharedPrefixPagedKVCacheWrapper {}

impl BatchDecodeWithSharedPrefixPagedKVCacheWrapper {
    pub fn new(workspace_buffer: UniqueTensor, kv_layout: Layout) -> Self {
        todo!()
    }

    pub fn begin_forward(
        indptr: &UniqueTensor,
        indices: &UniqueTensor,
        last_page_len: &UniqueTensor,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        data_type: String,
    ) {
    }

    pub fn forward(
        q: &UniqueTensor,
        k_shared: &UniqueTensor,
        v_shared: &UniqueTensor,
        unique_kv_data: &UniqueTensor,
        allow_fp16_qk_reduction: bool,

        sm_scale: Option<f32>,
        rope_scale: Option<f32>,
        rope_theta: Option<f32>,
    ) -> UniqueTensor {
        todo!()
    }

    pub fn end_forward() {}

    pub fn reset_workspace_buffer(new_workspace_buffer: &UniqueTensor) {
        todo!()
    }
}

pub fn cascade_attn_varlen(
    q: &UniqueTensor,
    k: &UniqueTensor,
    v: &UniqueTensor,
    seqlens_q: &UniqueTensor,
    seqlens_k: &UniqueTensor,
    max_seqlen_q: u32,
    max_seqlen_k: u32,
    softmax_scale: f32,
    causal: bool,
) -> UniqueTensor {
    todo!()
}

//
// use candle::backend::BackendStorage;
// use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
// use candle::cuda_backend::cudarc::driver::DevicePtr;
// use candle::cuda_backend::WrapErr;
// use candle::{CpuStorage, Layout, Result, Shape, Tensor};
// use half::{bf16, f16};
// use std::ptr;
//
// struct FlashAttnVarLen {
//     softmax_scale: f32,
//     causal: bool,
//     max_seqlen_q: u32,
//     max_seqlen_k: u32,
//     seqlens_q: Tensor,
//     seqlens_k: Tensor,
// }
//
// fn round_multiple(x: u32, m: u32) -> u32 {
//     (x + m - 1) / m * m
// }
//
// impl FlashAttnVarLen {
//     fn cuda_fwd_t<
//         T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
//     >(
//         &self,
//         q: &candle::CudaStorage,
//         q_l: &Layout,
//         k: &candle::CudaStorage,
//         k_l: &Layout,
//         v: &candle::CudaStorage,
//         v_l: &Layout,
//         is_bf16: bool,
//     ) -> Result<(candle::CudaStorage, Shape)> {
//         // https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/csrc/flash_attn/flash_api.cpp#L327
//         let dev = q.device();
//         let out_shape = q_l.shape().clone();
//         let out_l = Layout::contiguous(&out_shape);
//
//         let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
//         let seqlens_q = match &*seqlens_q {
//             candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be u32!
//             _ => candle::bail!("seqlens_q must be a cuda tensor"),
//         };
//         let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
//             Some((o1, o2)) => seqlens_q.slice(o1..o2),
//             None => candle::bail!("seqlens_q has to be contiguous"),
//         };
//
//         let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
//         let seqlens_k = match &*seqlens_k {
//             candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be u32!
//             _ => candle::bail!("seqlens_k must be a cuda tensor"),
//         };
//         let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
//             Some((o1, o2)) => seqlens_k.slice(o1..o2),
//             None => candle::bail!("seqlens_k has to be contiguous"),
//         };
//
//         let q = q.as_cuda_slice::<T>()?;
//         let k = k.as_cuda_slice::<T>()?;
//         let v = v.as_cuda_slice::<T>()?;
//         let q = q.slice(q_l.start_offset()..);
//         let k = k.slice(k_l.start_offset()..);
//         let v = v.slice(v_l.start_offset()..);
//
//         let q_stride = q_l.stride();
//         let k_stride = k_l.stride();
//         let v_stride = v_l.stride();
//         let o_stride = out_l.stride();
//
//         let q_rank = q_stride.len();
//         let k_rank = k_stride.len();
//         let v_rank = v_stride.len();
//         let o_rank = o_stride.len();
//
//         if q_rank != 3 || k_rank != 3 || v_rank != 3 {
//             candle::bail!(
//                 "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
//             )
//         }
//         if q_stride[q_rank - 1] != 1 {
//             candle::bail!("the last dim of q must be contiguous {q_stride:?}")
//         }
//         if k_stride[k_rank - 1] != 1 {
//             candle::bail!("the last dim of k must be contiguous {k_stride:?}")
//         }
//         if v_stride[v_rank - 1] != 1 {
//             candle::bail!("the last dim of v must be contiguous {v_stride:?}")
//         }
//
//         let (total_q, num_heads, head_size) = q_l.shape().dims3()?;
//         let (total_k, num_heads_k, _head_size) = k_l.shape().dims3()?;
//         let expected_kv = (total_k, num_heads_k, head_size);
//         if expected_kv != k_l.shape().dims3()? {
//             candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
//         }
//         if expected_kv != v_l.shape().dims3()? {
//             candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
//         }
//         if head_size > 256 {
//             candle::bail!("only supports head dimension at most 256 (got {head_size})")
//         }
//         if head_size % 8 != 0 {
//             // TODO: Handle head sizes that are not a multiple of 8 via some padding.
//             candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size})")
//         }
//         if num_heads % num_heads_k != 0 {
//             candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
//         }
//
//         let nseqlens_q = seqlens_q_layout.shape().dims1()?;
//         if nseqlens_q < 2 {
//             candle::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
//         }
//         let nseqlens_k = seqlens_k_layout.shape().dims1()?;
//         if nseqlens_k != nseqlens_q {
//             candle::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
//         }
//         let batch_size = nseqlens_q - 1;
//
//         let elem_count = out_shape.elem_count();
//         let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
//         let softmax_lse = dev
//             .alloc_zeros::<f32>(batch_size * num_heads * self.max_seqlen_q)
//             .w()?;
//
//         let blocksize_c = if head_size > 64 { 128 } else { 256 };
//         let max_seqlen_k_rounded = round_multiple(self.max_seqlen_k, blocksize_c);
//         let max_seqlen_q_rounded = round_multiple(self.max_seqlen_q, 16);
//
//         let dst_temp = if max_seqlen_k_rounded > blocksize_c {
//             Some(unsafe { dev.alloc::<f32>(total_q * num_heads * head_size) }.w()?)
//         } else {
//             None
//         };
//
//         let causal = if self.causal { 1 } else { 0 };
//         let is_bf16 = if is_bf16 { 1 } else { 0 };
//
//         let multi_processor_count = dev
//             .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
//             .w()?;
//
//         unsafe {
//             let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
//             let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
//             let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
//             let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
//             let dst_tmp_ptr = if let Some(slice) = &dst_temp {
//                 *slice.device_ptr() as *const core::ffi::c_void
//             } else {
//                 ptr::null()
//             };
//             let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
//             let seqlens_q_ptr = *seqlens_q.device_ptr() as *const core::ffi::c_int;
//             let seqlens_k_ptr = *seqlens_k.device_ptr() as *const core::ffi::c_int;
//             ffi::run_mha(
//                 q_ptr,
//                 k_ptr,
//                 v_ptr,
//                 dst_ptr,
//                 dst_tmp_ptr,
//                 softmax_lse_ptr,
//                 /* cu_seqlens_q_ptr */ seqlens_q_ptr,
//                 /* cu_seqlens_k_ptr */ seqlens_k_ptr,
//                 /* q_row_stride   */ q_stride[q_rank - 3] as u32,
//                 /* k_row_stride   */ k_stride[k_rank - 3] as u32,
//                 /* v_row_stride   */ v_stride[v_rank - 3] as u32,
//                 /* o_row_stride   */ o_stride[o_rank - 3] as u32,
//                 /* o_tmp_row_stride */ (num_heads * head_size) as u32,
//                 /* q_head_stride  */ q_stride[q_rank - 2] as u32,
//                 /* k_head_stride  */ k_stride[k_rank - 2] as u32,
//                 /* v_head_stride  */ v_stride[v_rank - 2] as u32,
//                 /* o_head_stride  */ o_stride[o_rank - 2] as u32,
//                 /* o_tmp_head_stride  */ head_size as u32,
//                 /* b */ batch_size as u32,
//                 /* h */ num_heads as u32,
//                 /* d */ head_size as u32,
//                 /* softmax_scale*/ self.softmax_scale,
//                 /* seqlen_q */ max_seqlen_q_rounded as u32,
//                 /* seqlen_k */ max_seqlen_k_rounded as u32,
//                 /* is_causal */ causal,
//                 /* is_bf16 */ is_bf16,
//                 /* multi_processor_count */ multi_processor_count,
//                 /* num_splits */ 0,
//             )
//         }
//
//         let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
//         Ok((dst, out_shape))
//     }
// }
//
// impl candle::CustomOp3 for FlashAttnVarLen {
//     fn name(&self) -> &'static str {
//         "flash-attn-varlen"
//     }
//
//     fn cpu_fwd(
//         &self,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//     ) -> Result<(CpuStorage, Shape)> {
//         candle::bail!("no cpu support for flash-attn")
//     }
//
//     fn cuda_fwd(
//         &self,
//         q: &candle::CudaStorage,
//         q_l: &Layout,
//         k: &candle::CudaStorage,
//         k_l: &Layout,
//         v: &candle::CudaStorage,
//         v_l: &Layout,
//     ) -> Result<(candle::CudaStorage, Shape)> {
//         match q.dtype() {
//             candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
//             candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
//             dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
//         }
//     }
// }
//
// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with variable-length batching.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
// /// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
// /// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
// /// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
// /// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
// /// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
// /// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
// ///
// /// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
// /// `seqlen_1 + seqlen_2`, etc.
// ///
// /// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
// pub fn flash_attn_varlen(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     seqlens_q: &Tensor,
//     seqlens_k: &Tensor,
//     max_seqlen_q: u32,
//     max_seqlen_k: u32,
//     softmax_scale: f32,
//     causal: bool,
// ) -> Result<Tensor> {
//     let op = FlashAttnVarLen {
//         softmax_scale,
//         causal,
//         max_seqlen_q,
//         max_seqlen_k,
//         seqlens_q: seqlens_q.clone(),
//         seqlens_k: seqlens_k.clone(),
//     };
//     q.apply_op3(k, v, op)
// }

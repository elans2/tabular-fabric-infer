#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use arrow::array::{Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use anyhow::{bail, Context};

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use itertools::Itertools;
use llama_cpp_2::context::LlamaContext;
use tokenizers::Tokenizer;

use structmap::FromMap;
use structmap_derive::FromMap;

use tracing::{error, info};

use crate::base::{InferContext, ModelInfer};
use crate::errors::InferError;
use crate::models::constants::RESULT_COLUMN_NAME;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

pub fn device(cpu: bool) -> Result<Device, InferError> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            info!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

#[derive(FromMap)]
struct GgmlLLamaArg {
    /// Run on CPU rather than on GPU.
    cpu: bool,

    use_flash_attn: bool,

    /// The temperature used to generate samples.
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    top_p: f64,

    /// The seed to use when generating random samples.
    seed: u64,

    sample_len: u64,

    model_file: String,

    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f64,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: u64,
}

impl Default for GgmlLLamaArg {
    fn default() -> Self {
        Self {
            model_file: "".to_string(),
            cpu: true,
            use_flash_attn: false,
            temperature: 0.0,
            top_p: 100 as f64,
            seed: 299792458,
            sample_len: 500,
            quantized: true,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

pub struct GgmlLLamaModelInfer {
    backend: Arc<RefCell<Option<LlamaBackend>>>,
    model: Arc<RefCell<Option<LlamaModel>>>,
}

impl GgmlLLamaModelInfer {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(RefCell::new(None)),
            model: Arc::new(RefCell::new(None)),
        }
    }
}

unsafe impl Sync for GgmlLLamaModelInfer {}
unsafe impl Send for GgmlLLamaModelInfer {}

impl ModelInfer for GgmlLLamaModelInfer {

    fn file_resources(&self) -> Vec<String> {
        vec!["model_file".to_string()]
    }

    fn load(
        &mut self,
        options: HashMap<String, String>,
    ) -> Result<bool, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = crate::models::ggml::GgmlLLamaArg::from_stringmap(arg);
        info!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            arg.temperature, arg.repeat_penalty, arg.repeat_last_n
        );

        let model_file = arg.model_file.clone();

        let backend = LlamaBackend::init().unwrap();

        // offload all layers to the gpu
        let model_params = {
            #[cfg(feature = "cuda")]
            if !arg.cpu {
                LlamaModelParams::default().with_n_gpu_layers(1000)
            } else {
                LlamaModelParams::default()
            }
            #[cfg(not(feature = "cuda"))]
            LlamaModelParams::default()
        };

        let mut model_params = pin!(model_params);

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        let model_path = PathBuf::from_str(model_file.as_str()).unwrap();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| "unable to load model")?;

        let mut self_backend = self.backend.clone();
        let mut self_backend = self_backend.borrow_mut();
        self_backend.get_or_insert(backend);

        let mut self_model = self.model.clone();
        let mut self_model = self_model.borrow_mut();
        self_model.get_or_insert(model);
        //
        // // initialize the context
        // let mut ctx_params = LlamaContextParams::default()
        //     .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        //     .with_seed(arg.seed as u32);
        //
        // ctx_params = ctx_params.with_n_threads(arg.threads as u32);
        // ctx_params = ctx_params.with_n_threads_batch(arg.threads_batch as u32);
        //
        // let mut ctx = model2.clone()
        //     .new_context(&backend, ctx_params)
        //     .with_context(|| "unable to create the llama_context").unwrap();
        //
        // //self_backend.get_or_insert(backend);
        // let mut self_pipeline = self.pipeline.clone();
        // let mut self_pipeline = self_pipeline.borrow_mut();
        // self_pipeline.get_or_insert(ctx);

        Ok(true)
    }

    fn infer(
        &mut self,
        batch: &RecordBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = crate::models::ggml::GgmlLLamaArg::from_stringmap(arg);

        let array = batch.column(0);
        let values = array.as_any().downcast_ref::<StringArray>().unwrap();

        let mut backend = self.backend.clone();
        let mut backend = backend.borrow_mut();
        let mut backend = backend.as_mut().unwrap();

        let mut model = self.model.clone();
        let mut model = model.borrow_mut();
        let mut model = model.as_mut().unwrap();

        // // initialize the context
        let mut ctx_params = LlamaContextParams::default().with_seed(arg.seed as u32);

        let mut ctx = model
            .new_context(&backend, ctx_params)
            .with_context(|| "unable to create the llama_context").unwrap();

        let mut result_values: Vec<String> = vec![];
        for value in values.iter() {
            ctx.clear_kv_cache();
            let prompt = value.unwrap();

            let mut new_tokens = vec![];

            let tokens_list = model
                .str_to_token(prompt, AddBos::Always)
                .with_context(|| format!("failed to tokenize {prompt}")).unwrap();

            let n_len = tokens_list.len() + arg.sample_len as usize;
            if tokens_list.len() >= usize::try_from(n_len).unwrap() {
                error!("the prompt is too long, it has more tokens than n_len")
            }

            for token in &tokens_list {
                info!("{}", model.token_to_str(*token, Special::Plaintext).unwrap());
            }

            // create a llama_batch with size 512
            // we use this object to submit token data for decoding
            let mut batch = LlamaBatch::new(512, 1);

            let last_index: i32 = (tokens_list.len() - 1) as i32;
            for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
                // llama_decode will output logits only for the last token of the prompt
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last).unwrap();
            }

            ctx.decode(&mut batch)
                .with_context(|| "llama_decode() failed").unwrap();

            let mut n_cur = batch.n_tokens();
            let mut n_decode = 0;

            // The `Decoder`
            let mut decoder = encoding_rs::UTF_8.new_decoder();


            while n_cur <= n_len as i32 {
                // sample the next token
                {
                    let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

                    let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                    // sample the most likely token
                    let new_token_id = ctx.sample_token_greedy(candidates_p);

                    // is it an end of stream?
                    if new_token_id == model.token_eos() {
                        break;
                    }

                    let output_bytes = model.token_to_bytes(new_token_id, Special::Plaintext).unwrap();
                    // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                    let mut output_string = String::with_capacity(32);
                    let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                    new_tokens.push(output_string);

                    batch.clear();
                    batch.add(new_token_id, n_cur, &[0], true).unwrap();
                }

                n_cur += 1;

                ctx.decode(&mut batch).with_context(|| "failed to eval").unwrap();

                n_decode += 1;
            }
            let new_text = new_tokens.join("");
            result_values.push(new_text);
        }

        println!("result_values: {:#?}", result_values);

        let result_batch = RecordBatch::try_from_iter(vec![(
            RESULT_COLUMN_NAME,
            Arc::new(StringArray::from(result_values)) as _,
        )])
            .unwrap();
        Ok(result_batch)
    }
}

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use arrow::array::{Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use itertools::Itertools;
use tokenizers::Tokenizer;

use structmap::FromMap;
use structmap_derive::FromMap;

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use crate::base::{InferContext, ModelInfer};
use crate::errors::InferError;

pub fn device(cpu: bool) -> Result<Device, InferError> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
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

    /// The length of the sample to generate (in tokens).
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
            sample_len: 100,
            quantized: true,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

pub struct GgmlLLamaModelInfer {
    pipeline: Arc<RefCell<Option<LLama>>>,
}

impl GgmlLLamaModelInfer {
    pub fn new() -> Self {
        Self {
            pipeline: Arc::new(RefCell::new(None)),
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
        &self,
        options: HashMap<String, String>,
    ) -> Result<bool, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = crate::models::ggml::GgmlLLamaArg::from_stringmap(arg);
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            arg.temperature, arg.repeat_penalty, arg.repeat_last_n
        );

        let model_file = arg.model_file.clone();
        let mut model_options = ModelOptions::default();
        //model_options.n_gpu_layers = 33;
        //model_options.main_gpu = "cuda".to_string();
        println!("{}", model_file);
        println!("{:#?}", model_options);
        let llama = LLama::new(model_file.as_str().into(), &model_options).unwrap();

        let mut self_pipeline = self.pipeline.clone();
        let mut self_pipeline = self_pipeline.borrow_mut();
        self_pipeline.get_or_insert(llama);

        Ok(true)
    }

    fn infer(
        &self,
        batch: &RecordBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferError> {
        //let result = pipeline.run(&args.prompt, args.sample_len)?;
        let array = batch.column(0);
        let values = array.as_any().downcast_ref::<StringArray>().unwrap();
        let mut pipeline = self.pipeline.clone();
        let mut pipeline = pipeline.borrow_mut();
        let mut pipeline = pipeline.as_mut().unwrap();

        let mut result_values = vec![];
        for value in values.iter() {
            let value = value.unwrap();

            let result_tokens = Arc::new(Mutex::new(RefCell::new(vec![])));
            let callback_result_tokens = result_tokens.clone();
            let predict_options = PredictOptions {
                tokens: 0,
                threads: 4,
                top_k: 90,
                top_p: 0.86,
                token_callback: Some(Box::new(move |token| {
                    callback_result_tokens
                        .lock()
                        .unwrap()
                        .borrow_mut()
                        .push(token);
                    true
                })),
                ..Default::default()
            };

            pipeline
                .predict(value.to_string(), predict_options)
                .unwrap();
            let result_value = result_tokens.lock().unwrap().borrow().join("");
            result_values.push(result_value);
        }

        let result_batch = RecordBatch::try_from_iter(vec![(
            "col",
            Arc::new(StringArray::from(result_values)) as _,
        )])
            .unwrap();

        Ok(result_batch)
    }
}

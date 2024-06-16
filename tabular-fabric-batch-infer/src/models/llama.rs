#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use candle_transformers::models::llama;
use candle_transformers::models::llama::{Config, Llama as Model, LlamaConfig};

use crate::base::{InferBatch, InferContext, ModelInfer};
use crate::errors::InferError;
use crate::models::candle_gen;
use crate::models::candle_gen::{BatchGen, BatchGenModel};
use crate::models::constants::RESULT_COLUMN_NAME;
use crate::models::phi::CandlePhiTextGenModel;
use crate::models::qwen::CandleQwenTextGenModel;
use crate::models::utils::token_output_stream::TokenOutputStream;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use itertools::Itertools;
use structmap::FromMap;
use structmap_derive::FromMap;
use tokenizers::Tokenizer;
use tracing::{debug, info};

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
struct CandleLlamaArg {
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

    tokenizer_file: String,

    weight_files: String,

    config_file: String,

    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f64,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: u64,

    max_batch_size: u64,
}

impl Default for CandleLlamaArg {
    fn default() -> Self {
        Self {
            tokenizer_file: "".to_string(),
            weight_files: "".to_string(),
            config_file: "".to_string(),
            cpu: true,
            use_flash_attn: false,
            temperature: 0.0,
            top_p: 100 as f64,
            sample_len: 500,
            seed: 299792458,
            quantized: false,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            max_batch_size: 1,
        }
    }
}

pub struct CandleLlamaModelInfer {
    pipeline: Arc<RefCell<Option<CandleLlamaTextGen>>>,
}

unsafe impl Sync for CandleLlamaModelInfer {}
unsafe impl Send for CandleLlamaModelInfer {}

struct CacheBuilder {
    use_kv_cache: bool,
    dtype: DType,
    config: Config,
    device: Device,
}

impl CacheBuilder {
    fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Self {
        Self {
            use_kv_cache,
            dtype,
            config: config.clone(),
            device: device.clone(),
        }
    }

    fn build(&self) -> llama::Cache {
        llama::Cache::new(self.use_kv_cache, self.dtype, &self.config, &self.device).unwrap()
    }
}

impl CandleLlamaModelInfer {
    pub fn new() -> Self {
        Self {
            pipeline: Arc::new(RefCell::new(None)),
        }
    }
}

impl ModelInfer for CandleLlamaModelInfer {
    fn file_resources(&self) -> Vec<String> {
        vec!["tokenizer_file".to_string(), "weight_files".to_string()]
    }

    fn load(&mut self, options: HashMap<String, String>) -> Result<bool, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandleLlamaArg::from_stringmap(arg);

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

        let tokenizer_file = std::path::PathBuf::from(arg.tokenizer_file);

        let weight_files = arg
            .weight_files
            .split(",")
            .map(|x| std::path::PathBuf::from(x))
            .collect::<Vec<std::path::PathBuf>>();

        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let tokenizers = (0..arg.max_batch_size)
            .map(|_| tokenizer.clone())
            .collect::<Vec<_>>();

        let mut config: LlamaConfig =
            serde_json::from_slice(&std::fs::read(arg.config_file).unwrap()).unwrap();
        let config = config.into_config(arg.use_flash_attn);
        let (model, device, cache_builder) = if arg.quantized {
            unimplemented!()
        } else {
            let device = device(arg.cpu)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
            let model = Model::load(vb, &config)?;

            let use_kv_cache = true;
            let builder = CacheBuilder::new(use_kv_cache, dtype, &config, &device);

            (ModelMode::Normal(model), device, builder)
        };

        let mut pipeline = CandleLlamaTextGen::new(
            model,
            cache_builder,
            tokenizers,
            arg.seed,
            Some(arg.temperature),
            Some(arg.top_p),
            arg.repeat_penalty as f32,
            arg.repeat_last_n as usize,
            &device,
        );

        let mut self_pipeline = self.pipeline.clone();
        let mut self_pipeline = self_pipeline.borrow_mut();
        self_pipeline.get_or_insert(pipeline);
        Ok(true)
    }

    fn infer(
        &mut self,
        batch: &InferBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<InferBatch, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandleLlamaArg::from_stringmap(arg);

        let first_column_name = batch.column_names().first().unwrap();
        let values = batch.column_values(first_column_name).unwrap();
        let mut pipeline = self.pipeline.clone();
        let mut pipeline = pipeline.borrow_mut();
        let mut pipeline = pipeline.as_mut().unwrap();

        let values = values.iter().collect::<Vec<_>>();
        let mut pos_list = vec![];
        let mut value_list = vec![];
        for (pos, value) in values.iter().enumerate() {
            if value.is_empty() {
                continue;
            }
            pos_list.push(pos);
            value_list.push(value.clone().to_string());
        }

        let mut gen_value_list = vec![];
        for chunk_values in &value_list.into_iter().chunks(arg.max_batch_size as usize) {
            let chunk_values = chunk_values.collect_vec();
            let mut chunk_result_values = pipeline
                .gen(&chunk_values, arg.sample_len as usize)
                .unwrap();
            gen_value_list.append(&mut chunk_result_values);
        }

        let mut result_value_list = vec![];
        for (pos, gen_value) in pos_list.iter().zip(gen_value_list.iter()) {
            for _ in 0..(pos - result_value_list.len()) {
                result_value_list.push("".to_string());
            }
            result_value_list.push(gen_value.to_owned().join(""));
        }
        let result_batch = InferBatch::new(
            batch.column_names(),
            &HashMap::from([(RESULT_COLUMN_NAME.to_string(), result_value_list.clone())]),
        )
        .unwrap();
        Ok(result_batch)
    }
}

enum ModelMode {
    Normal(Model),
}

pub struct CandleLlamaTextGen {
    model: CandleLlamaTextGenModel,
    device: Device,
    tokenizers: Vec<TokenOutputStream>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl CandleLlamaTextGen {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelMode,
        cache_builder: CacheBuilder,
        tokenizers: Vec<Tokenizer>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        let tokenizers = tokenizers
            .into_iter()
            .map(|t| TokenOutputStream::new(t))
            .collect::<Vec<_>>();
        let model = CandleLlamaTextGenModel::new(model, cache_builder);
        Self {
            model,
            tokenizers,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }
}

pub struct CandleLlamaTextGenModel {
    model: ModelMode,
    cache_builder: CacheBuilder,
    using_cache: llama::Cache,
}

impl CandleLlamaTextGenModel {
    pub fn new(model: ModelMode, cache_builder: CacheBuilder) -> Self {
        let using_cache = cache_builder.build();
        Self {
            model,
            cache_builder,
            using_cache,
        }
    }
}

impl BatchGenModel for CandleLlamaTextGenModel {
    fn forward(
        &mut self,
        batch_input: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor, InferError> {
        let logits = match &mut self.model {
            ModelMode::Normal(m) => {
                m.forward(&batch_input, seqlen_offset, &mut self.using_cache)?
            }
        };
        Ok(logits)
    }

    fn reset(&mut self) -> Result<(), InferError> {
        self.using_cache = self.cache_builder.build();
        Ok(())
    }
}

impl BatchGen for CandleLlamaTextGen {
    fn gen(
        &mut self,
        prompts: &Vec<String>,
        sample_len: usize,
    ) -> Result<Vec<Vec<String>>, InferError> {
        println!("prompts: {:#?}", prompts);
        for tokenizer in &mut self.tokenizers {
            tokenizer.clear();
        }
        let eos_token = match self.tokenizers[0].get_token("<|end_of_text|>") {
            Some(token) => token,
            None => {
                return Err(InferError::GenericError {
                    msg: "cannot find the </s> token".to_string(),
                })
            }
        };
        let pad_token = match self.tokenizers[0].get_token("<|end_of_text|>") {
            Some(token) => token,
            None => {
                return Err(InferError::GenericError {
                    msg: "cannot find the <unk> token".to_string(),
                })
            }
        };

        self.model.reset()?;
        let result_batch_tokens = candle_gen::batch_infer(
            &mut self.model,
            &mut self.tokenizers,
            &mut self.logits_processor,
            prompts.clone(),
            sample_len,
            &self.device,
            eos_token,
            pad_token,
        )
        .unwrap();
        Ok(result_batch_tokens)
    }
}

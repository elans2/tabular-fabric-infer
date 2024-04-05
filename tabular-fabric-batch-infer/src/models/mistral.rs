#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use arrow::array::{Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use candle_transformers::models::mistral::{Config, Model};
use candle_transformers::models::quantized_mistral::Model as QModel;

use crate::errors::InferError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use crate::base::{ModelInfer, InferContext};
use crate::models::utils::token_output_stream::TokenOutputStream;
use structmap::FromMap;
use structmap_derive::FromMap;
use tracing::{debug, info};
use crate::models::constants::RESULT_COLUMN_NAME;
use crate::models::ggml::GgmlLLamaModelInfer;
use itertools::Itertools;
use crate::models::candle_gen;
use crate::models::candle_gen::{BatchGen, BatchGenModel};
use crate::models::phi::CandlePhiTextGenModel;
use crate::models::qwen::CandleQwenTextGenModel;

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
struct CandleMistralArg {
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

    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f64,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: u64,

    max_batch_size: u64,
}

impl Default for CandleMistralArg {
    fn default() -> Self {
        Self {
            tokenizer_file: "".to_string(),
            weight_files: "".to_string(),
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

pub struct CandleMistralModelInfer {
    pipeline: Arc<RefCell<Option<CandleMistralTextGen>>>,
}

unsafe impl Sync for CandleMistralModelInfer {}
unsafe impl Send for CandleMistralModelInfer {}

impl CandleMistralModelInfer {
    pub fn new() -> Self {
        Self {
            pipeline: Arc::new(RefCell::new(None)),
        }
    }
}

impl ModelInfer for CandleMistralModelInfer {

    fn file_resources(&self) -> Vec<String> {
        vec![
            "tokenizer_file".to_string(),
            "weight_files".to_string(),
        ]
    }

    fn load(
        &mut self,
        options: HashMap<String, String>,
    ) -> Result<bool, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandleMistralArg::from_stringmap(arg);

        info!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        info!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            arg.temperature,
            arg.repeat_penalty,
            arg.repeat_last_n
        );

        let tokenizer_file = std::path::PathBuf::from(
            arg.tokenizer_file,
        );

        let weight_files = arg.weight_files.split(",").map(|x| std::path::PathBuf::from(x)).collect::<Vec<std::path::PathBuf>>();

        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let tokenizers = (0..arg.max_batch_size).map(|_| tokenizer.clone()).collect::<Vec<_>>();

        let config = Config::config_7b_v0_1(arg.use_flash_attn);
        let (model, device) = if arg.quantized {
            let filename = &weight_files[0];
            let device = Device::new_metal(0)?;
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
            let model = QModel::new(&config, vb)?;
            (ModelMode::Quantized(model), Device::Cpu)
        } else {
            let device = device(arg.cpu)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
            let model = Model::new(&config, vb)?;
            (ModelMode::Normal(model), device)
        };

        let mut pipeline = CandleMistralTextGen::new(
            model,
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
        batch: &RecordBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandleMistralArg::from_stringmap(arg);

        let array = batch.column(0);
        let values = array.as_any().downcast_ref::<StringArray>().unwrap();
        let mut pipeline = self.pipeline.clone();
        let mut pipeline = pipeline.borrow_mut();
        let mut pipeline = pipeline.as_mut().unwrap();

        let values = values.iter().collect::<Vec<_>>();
        let mut pos_list = vec![];
        let mut value_list = vec![];
        for (pos, value) in values.iter().enumerate() {
            if value.is_none() {
                continue;
            }
            if let Some(value) = value {
                pos_list.push(pos);
                value_list.push(value.clone().to_string());
            }
        }

        let mut gen_value_list = vec![];
        for chunk_values in &value_list.into_iter().chunks(100) {
            let chunk_values= chunk_values.collect_vec();
            let mut chunk_result_values = pipeline.gen(&chunk_values, arg.sample_len as usize).unwrap();
            gen_value_list.append(&mut chunk_result_values);
        }

        let mut result_value_list = vec![];
        for (pos, gen_value) in pos_list.iter().zip(gen_value_list.iter()) {
            for _ in 0 .. (pos - result_value_list.len()) {
                result_value_list.push("".to_string());
            }
            result_value_list.push(gen_value.to_owned().join(""));
        }
        let result_batch = RecordBatch::try_from_iter(vec![(
            RESULT_COLUMN_NAME,
            Arc::new(StringArray::from(result_value_list)) as _,
        )]).map_err(|e| InferError::ArrowError { source: e })?;
        Ok(result_batch)
    }
}

enum ModelMode {
    Normal(Model),
    Quantized(QModel),
}

pub struct CandleMistralTextGen {
    model: CandleMistralTextGenModel,
    device: Device,
    tokenizers: Vec<TokenOutputStream>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl CandleMistralTextGen {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelMode,
        tokenizers: Vec<Tokenizer>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        let tokenizers = tokenizers.into_iter().map(|t| TokenOutputStream::new(t)).collect::<Vec<_>>();
        let model = CandleMistralTextGenModel::new(model);
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

pub struct CandleMistralTextGenModel {
    model: ModelMode
}

impl crate::models::mistral::CandleMistralTextGenModel {
    pub fn new(model: ModelMode) -> Self {
        Self {
            model
        }
    }
}

impl BatchGenModel for crate::models::mistral::CandleMistralTextGenModel {
    fn forward(&mut self, batch_input: &Tensor) -> Result<Tensor, InferError> {
        let seq_len = batch_input.shape().dims()[1];
        let logits = match &mut self.model {
            ModelMode::Normal(m) => m.forward(&batch_input, seq_len)?,
            ModelMode::Quantized(m) => m.forward(&batch_input, seq_len)?,
        };
        Ok(logits)
    }

    fn reset(&mut self) -> Result<(), InferError> {
        match &mut self.model {
            ModelMode::Normal(m) => m.clear_kv_cache(),
            ModelMode::Quantized(m) => m.clear_kv_cache(),
        };
        Ok(())
    }
}

impl BatchGen for CandleMistralTextGen {
    fn gen(&mut self, prompts: &Vec<String>, sample_len: usize) -> Result<Vec<Vec<String>>, InferError> {
        println!("prompts: {:#?}", prompts);
        for tokenizer in &mut self.tokenizers {
            tokenizer.clear();
        }
        let eos_token = match self.tokenizers[0].get_token("<|endoftext|>") {
            Some(token) => token,
            None => {
                return Err(InferError::GenericError {
                    msg: "cannot find the </s> token".to_string(),
                })
            }
        };
        let pad_token = match self.tokenizers[0].get_token("<|endoftext|>") {
            Some(token) => token,
            None => {
                return Err(InferError::GenericError {
                    msg: "cannot find the <unk> token".to_string(),
                })
            }
        };

        self.model.reset()?;
        let result_batch_tokens = candle_gen::batch_infer(&mut self.model, &mut self.tokenizers, &mut self.logits_processor, prompts.clone(), sample_len, &self.device, eos_token, pad_token).unwrap();
        Ok(result_batch_tokens)
    }
}

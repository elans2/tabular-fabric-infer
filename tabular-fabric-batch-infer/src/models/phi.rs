#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use arrow::array::{Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::cmp::max;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::Arc;

use candle_transformers::models::phi::{Config, Model};

use crate::errors::InferError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use itertools::Itertools;
use tokenizers::Tokenizer;
use crate::base::{ModelInfer, InferContext};
use crate::models::utils::token_output_stream::TokenOutputStream;
use structmap::FromMap;
use structmap_derive::FromMap;
use tracing::{debug, info};
use crate::models::candle_gen::{BatchGen, BatchGenModel};
use crate::models::constants::RESULT_COLUMN_NAME;
use crate::models::ggml::GgmlLLamaModelInfer;
use crate::models::mistral::CandleMistralModelInfer;
use crate::models::{candle_gen, utils};

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

#[derive(FromMap, Clone)]
struct CandlePhiArg {
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

impl Default for CandlePhiArg {
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

pub struct CandlePhiModelInfer {
    pipeline: Arc<RefCell<Option<CandlePhiTextGen>>>,
    arg: Option<CandlePhiArg>,
}

unsafe impl Sync for CandlePhiModelInfer {}
unsafe impl Send for CandlePhiModelInfer {}

impl CandlePhiModelInfer {
    pub fn new() -> Self {
        Self {
            pipeline: Arc::new(RefCell::new(None)),
            arg: None,
        }
    }
}

impl ModelInfer for CandlePhiModelInfer {

    fn file_resources(&self) -> Vec<String> {
        vec![
            "tokenizer_file".to_string(),
            "config_file".to_string(),
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
        let arg = CandlePhiArg::from_stringmap(arg);

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

        self.arg = Some(arg.clone());
        let tokenizer_file = std::path::PathBuf::from(
            arg.tokenizer_file,
        );

        let weight_files = arg.weight_files.split(",").map(|x| std::path::PathBuf::from(x)).collect::<Vec<std::path::PathBuf>>();

        info!("load tokenizer file {:?}", tokenizer_file);
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let tokenizers = (0..arg.max_batch_size).map(|_| tokenizer.clone()).collect::<Vec<_>>();

        let config = std::fs::read_to_string(arg.config_file).unwrap();
        let config: Config = serde_json::from_str(&config).unwrap();
        let (model, device) = if arg.quantized {
            todo!()
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

        let mut pipeline = CandlePhiTextGen::new(
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
        println!("start infer 0");
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandlePhiArg::from_stringmap(arg);

        let array = batch.column(0);
        let values = array.as_any().downcast_ref::<StringArray>().unwrap();
        let mut pipeline = self.pipeline.clone();
        let mut pipeline = pipeline.borrow_mut();
        let mut pipeline = pipeline.as_mut().unwrap();

        let values = values.iter().collect::<Vec<_>>();
        let (mut pos_list, mut value_list) = candle_gen::filter_not_empty_values(&values).unwrap();

        println!("start infer 1");

        println!("pos list {:#?}", pos_list);
        println!("value list {:#?}", value_list);

        let mut gen_value_list = candle_gen::chunk_gen(pipeline, value_list, arg.max_batch_size as usize, arg.sample_len as usize).unwrap();
        let mut result_value_list = candle_gen::align_gen_values_with_position(pos_list, gen_value_list).unwrap();
        let result_batch = RecordBatch::try_from_iter(vec![(
            RESULT_COLUMN_NAME,
            Arc::new(StringArray::from(result_value_list)) as _,
        )]).map_err(|e| InferError::ArrowError { source: e })?;
        println!("{:#?}", result_batch);
        Ok(result_batch)
    }
}

enum ModelMode {
    Normal(Model),
}

pub struct CandlePhiTextGen {
    model: CandlePhiTextGenModel,
    device: Device,
    tokenizers: Vec<TokenOutputStream>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl CandlePhiTextGen {
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
        let model = CandlePhiTextGenModel::new(model);
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

pub struct CandlePhiTextGenModel {
    model: ModelMode
}

impl CandlePhiTextGenModel {
    pub fn new(model: ModelMode) -> Self {
        Self {
            model
        }
    }
}

impl BatchGenModel for CandlePhiTextGenModel {
    fn forward(&mut self, batch_input: &Tensor, seqlen_offset: usize) -> Result<Tensor, InferError> {
        println!("{:#?}, {:#?}", batch_input.shape(), batch_input.shape().dims()[1]);
        let logits = match &mut self.model {
            ModelMode::Normal(m) => m.forward(&batch_input)?,
        };
        Ok(logits)
    }

    fn reset(&mut self) -> Result<(), InferError> {
        match &mut self.model {
            ModelMode::Normal(m) => m.clear_kv_cache(),
        };
        Ok(())
    }
}

impl BatchGen for CandlePhiTextGen {

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

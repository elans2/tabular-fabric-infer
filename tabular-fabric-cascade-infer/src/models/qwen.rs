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

use candle_transformers::models::qwen2::{Config, Model};

use crate::errors::InferError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use crate::base::{GeneralInnerModelInfer, InferContext};
use crate::utils::token_output_stream::TokenOutputStream;
use structmap::FromMap;
use structmap_derive::FromMap;

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
struct CandleQwenArg {
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

    config_file: String,

    tokenizer_file: String,

    weight_files: String,

    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f64,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: u64,
}

impl Default for CandleQwenArg {
    fn default() -> Self {
        Self {
            config_file: "".to_string(),
            tokenizer_file: "".to_string(),
            weight_files: "".to_string(),
            cpu: true,
            use_flash_attn: false,
            temperature: 0.0,
            top_p: 100 as f64,
            seed: 299792458,
            sample_len: 100,
            quantized: false,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}



pub struct CandleQwenModelInfer {
    pipeline: Arc<RefCell<Option<CandleQwenTextGeneration>>>,
}

impl CandleQwenModelInfer {
    pub fn new() -> Self {
        Self {
            pipeline: Arc::new(RefCell::new(None)),
        }
    }
}

impl GeneralInnerModelInfer for CandleQwenModelInfer {
    fn load(
        &self,
        options: HashMap<String, String>,
    ) -> Result<bool, InferError> {
        let mut arg = StringMap::new();
        for kv in options.into_iter() {
            arg.insert(kv.0, kv.1);
        }
        let arg = CandleQwenArg::from_stringmap(arg);

        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            arg.temperature,
            arg.repeat_penalty,
            arg.repeat_last_n
        );

        println!("inner 1");
        let tokenizer_file = std::path::PathBuf::from(
            arg.tokenizer_file,
        );
        println!("inner 2");

        let weight_files = arg.weight_files.split(",").map(|x| std::path::PathBuf::from(x)).collect::<Vec<std::path::PathBuf>>();

        println!("inner 3");
        println!("tokenizer file {:?}", tokenizer_file);
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

        println!("inner 4");
        let config = std::fs::read_to_string(arg.config_file).unwrap();
        let config: Config = serde_json::from_str(&config).unwrap();
        println!("inner 5");
        let (model, device) = if arg.quantized {
            todo!()
        } else {
            println!("inner 7");
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

        println!("inner 8");
        let mut pipeline = CandleQwenTextGeneration::new(
            model,
            tokenizer,
            arg.seed,
            Some(arg.temperature),
            Some(arg.top_p),
            arg.repeat_penalty as f32,
            arg.repeat_last_n as usize,
            &device,
        );

        println!("inner 9");
        let mut self_pipeline = self.pipeline.clone();
        let mut self_pipeline = self_pipeline.borrow_mut();
        self_pipeline.get_or_insert(pipeline);
        println!("inner 10");
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
            let result_value = pipeline.gen(value, 100).unwrap();
            let result_value = result_value.join(" ");
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

enum ModelMode {
    Normal(Model),
}

pub struct CandleQwenTextGeneration {
    model: ModelMode,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl CandleQwenTextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelMode,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn gen(&mut self, prompt: &str, sample_len: usize) -> Result<Vec<String>, InferError> {
        match &mut self.model {
            ModelMode::Normal(m) => m.clear_kv_cache(),
        };
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => {
                return Err(InferError::GenericError {
                    msg: "cannot find the <|endoftext|> token".to_string(),
                })
            },
        };
        let mut result_tokens = vec![];
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            println!("tokens: {} context_size {}, start_pos: {}", tokens.len(), context_size, start_pos);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?;
            println!("input: {:?}", input.shape());
            let input = input.unsqueeze(0)?;
            println!("input: {:?}", input.shape());
            let input2 = input.clone();
            let input3 = Tensor::cat(&[input.clone(), input2], 0)?;
            println!("input: {:?}", input3.shape());
            let input = input3.clone();
            let logits = match &mut self.model {
                ModelMode::Normal(m) => m.forward(&input, start_pos)?,
            };
            println!("logits: {:?}", logits.shape());
            let logits = logits.get(0)?;
            println!("logits A: {:?}", logits.shape());
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            println!("logits B: {:?}", logits.shape());
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            println!("logits C: {:?}", logits.shape());
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(token) = self.tokenizer.next_token(next_token)? {
                result_tokens.push(token);
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            result_tokens.push(rest);
        }
        Ok(result_tokens)
    }
}

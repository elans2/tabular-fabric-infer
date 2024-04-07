use candle_core::{Device, DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use itertools::Itertools;
use crate::errors::InferError;
use crate::models::utils::token_output_stream::TokenOutputStream;

pub trait BatchGenModel {
    fn forward(&mut self, batch_input: &Tensor, seqlen_offset: usize) -> Result<Tensor, InferError>;
    fn reset(&mut self) -> Result<(), InferError>;
}

pub trait BatchGen {
    fn gen(&mut self, prompts: &Vec<String>, sample_len: usize) -> Result<Vec<Vec<String>>, InferError>;
}

pub fn filter_not_empty_values(values: &Vec<Option<&str>>) -> Result<(Vec<usize>, Vec<String>), InferError>{
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
    Ok((pos_list, value_list))
}

pub fn chunk_gen(batch_gen: &mut impl BatchGen, prompts: Vec<String>, chunk_size: usize, sample_len: usize) -> Result<Vec<Vec<String>>, InferError> {
    let mut gen_values = vec![];
    for chunk_prompts in &prompts.into_iter().chunks(chunk_size) {
        let chunk_values = chunk_prompts.collect_vec();
        let mut chunk_result_values = batch_gen.gen(&chunk_values, sample_len).unwrap();
        gen_values.append(&mut chunk_result_values);
    }
    Ok(gen_values)
}

pub fn align_gen_values_with_position(pos_list: Vec<usize>, gen_value_list: Vec<Vec<String>>) -> Result<Vec<String>, InferError> {
    let mut result_value_list = vec![];
    for (pos, gen_value) in pos_list.iter().zip(gen_value_list.iter()) {
        for _ in 0 .. (pos - result_value_list.len()) {
            result_value_list.push("".to_string());
        }
        result_value_list.push(gen_value.to_owned().join(""));
    }
    Ok(result_value_list)
}

pub fn batch_infer(batch_gen_model: &mut impl BatchGenModel, tokenizers: &mut Vec<TokenOutputStream>, logits_processor: &mut LogitsProcessor, prompts: Vec<String>, sample_len: usize, device: &Device, eos_token: u32, pad_token: u32) -> Result<Vec<Vec<String>>, InferError> {
    let mut batch_tokens = vec![];
    for (pos, prompt) in prompts.iter().enumerate() {
        println!("{}, {}", pos, prompt);
        let tokens = tokenizers[pos]
            .tokenizer()
            .encode(prompt.to_owned(), true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        batch_tokens.push(tokens);
    }
    let mut max_seq_len = batch_tokens.clone().into_iter().map(|ts| ts.len()).max();
    match max_seq_len {
        Some(max_seq_len) => {
            for tokens in &mut batch_tokens {
                for _ in 0..(max_seq_len - tokens.len()) {
                    tokens.push(pad_token);
                }
            }
        }
        None => {
            return Err(InferError::GenericError {msg: "failed to get max seq len".to_string()});
        }
    }
    let mut generated_tokens = 0usize;
    let mut result_batch_tokens: Vec<Vec<String>> = vec![];
    println!("sample_len, {}", sample_len);
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { batch_tokens[0].len() };
        let start_pos = batch_tokens[0].len().saturating_sub(context_size);
        let mut row_inputs = vec![];
        for batch_idx in 0..batch_tokens.len() {
            let ctxt = &batch_tokens[batch_idx][start_pos..];
            let row_input = Tensor::new(ctxt, device)?;
            println!("row_input 1: {:#?}", row_input.shape());
            let row_input = row_input.unsqueeze(0)?;
            println!("row_input 2: {:#?}", row_input.shape());
            row_inputs.push(row_input);
        }
        let batch_input = Tensor::cat(&row_inputs, 0)?;
        println!("batch_input 1: {:#?}", batch_input.shape());
        let logits = batch_gen_model.forward(&batch_input, start_pos)?;

        for batch_idx in 0..batch_tokens.len() {
            let logits = logits.get(batch_idx)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            // let logits = if self.repeat_penalty == 1. {
            //     logits
            // } else {
            //     let start_at = result_tokens[0].len().saturating_sub(self.repeat_last_n);
            //     candle_transformers::utils::apply_repeat_penalty(
            //         &logits,
            //         self.repeat_penalty,
            //         &result_tokens[0][start_at..],
            //     )?
            // };
            let next_token = logits_processor.sample(&logits)?;
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            batch_tokens[batch_idx].push(next_token);
            if let Some(t) = tokenizers[batch_idx].next_token(next_token)? {
                if index == 0 {
                    result_batch_tokens.push(vec![t]);
                } else {
                    result_batch_tokens[batch_idx].push(t);
                }
            }
        }
    }
    for batch_idx in 0..batch_tokens.len() {
        if let Some(rest) = tokenizers[batch_idx].decode_rest().map_err(anyhow::Error::msg)? {
            result_batch_tokens[batch_idx].push(rest);
        }
    }
    Ok(result_batch_tokens)
}
use std::collections::{HashMap, HashSet};
use crate::errors::InferError;
use crate::infer::utils::token_output_stream::TokenOutputStream;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use itertools::{Itertools, max, min};

pub trait BatchGenModel {
    fn forward(&mut self, batch_input: &Tensor, seqlen_offset: usize)
        -> Result<Tensor, InferError>;
    fn reset(&mut self) -> Result<(), InferError>;
}

pub trait BatchGen {
    fn gen(
        &mut self,
        prompts: &Vec<String>,
        sample_len: usize,
    ) -> Result<Vec<Vec<String>>, InferError>;
}

pub fn filter_not_empty_values(
    values: &Vec<String>,
) -> Result<(Vec<usize>, Vec<String>), InferError> {
    let mut pos_list = vec![];
    let mut value_list = vec![];
    for (pos, value) in values.iter().enumerate() {
        if value.is_empty() {
            continue;
        }
        pos_list.push(pos);
        value_list.push(value.clone());
    }
    Ok((pos_list, value_list))
}

pub fn chunk_gen(
    batch_gen: &mut impl BatchGen,
    prompts: Vec<String>,
    chunk_size: usize,
    sample_len: usize,
) -> Result<Vec<Vec<String>>, InferError> {
    let mut gen_values = vec![];
    for chunk_prompts in &prompts.into_iter().chunks(chunk_size) {
        let chunk_values = chunk_prompts.collect_vec();
        let mut chunk_result_values = batch_gen.gen(&chunk_values, sample_len).unwrap();
        gen_values.append(&mut chunk_result_values);
    }
    Ok(gen_values)
}

pub fn align_gen_values_with_position(
    pos_list: Vec<usize>,
    gen_value_list: Vec<Vec<String>>,
) -> Result<Vec<String>, InferError> {
    let mut result_value_list = vec![];
    for (pos, gen_value) in pos_list.iter().zip(gen_value_list.iter()) {
        for _ in 0..(pos - result_value_list.len()) {
            result_value_list.push("".to_string());
        }
        result_value_list.push(gen_value.to_owned().join(""));
    }
    Ok(result_value_list)
}

pub fn batch_infer(
    batch_gen_model: &mut impl BatchGenModel,
    tokenizers: &mut Vec<TokenOutputStream>,
    logits_processor: &mut LogitsProcessor,
    prompts: Vec<String>,
    sample_len: usize,
    device: &Device,
    eos_token: u32,
    pad_token: u32,
) -> Result<Vec<Vec<String>>, InferError> {
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
    println!("batch tokens {:#?}", batch_tokens);
    let mut result_batch_tokens: Vec<Vec<String>> = vec![];
    println!("sample_len, {}", sample_len);

    let prompt_batch_tokens_lens = batch_tokens.iter().map(|b| b.len()).collect::<Vec<usize>>();
    let prompt_row_tokens_len_min = min(prompt_batch_tokens_lens.clone()).unwrap();
    let prompt_row_tokens_len_max = max(prompt_batch_tokens_lens.clone()).unwrap();
    let batch_tokens_max_lens = batch_tokens.iter().map(|b| b.len() + sample_len).collect::<Vec<usize>>();
    
    println!("{:#?}, {}, {}, {:#?}", prompt_batch_tokens_lens, prompt_row_tokens_len_min, prompt_row_tokens_len_max, batch_tokens_max_lens);

    let mut end_batches = HashSet::new();
    
    for sample_pos in 0..(prompt_row_tokens_len_max - prompt_row_tokens_len_min + sample_len) {
        let (start_pos, end_pos) = if sample_pos > 0 { (prompt_row_tokens_len_min + sample_pos - 1, prompt_row_tokens_len_min + sample_pos) } else { (0, prompt_row_tokens_len_min) };
        println!("sample pos {}, start_ps, end_pos {}, {}", sample_pos, start_pos, end_pos);
        let mut row_inputs = vec![];
        for batch_pos in 0..batch_tokens.len() {
            let ctxt = &batch_tokens[batch_pos][start_pos..end_pos];
            let row_input = Tensor::new(ctxt, device)?;
            println!("row_input 1: {:#?}", row_input.shape());
            let row_input = row_input.unsqueeze(0)?;
            println!("row_input 2: {:#?}", row_input.shape());
            row_inputs.push(row_input);
        }
        let batch_input = Tensor::cat(&row_inputs, 0)?;
        println!("batch_input 1: {:#?}", batch_input.shape());
        let logits = batch_gen_model.forward(&batch_input, start_pos)?;

        for batch_pos in 0..row_inputs.len() {
            let logits = logits.get(batch_pos)?;
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
            println!("batch pos {}", batch_pos);
            let next_token = logits_processor.sample(&logits)?;
            if sample_pos >= (prompt_batch_tokens_lens[batch_pos] - prompt_row_tokens_len_min) {
                batch_tokens[batch_pos].push(next_token);
            }
            println!("batch pos {}, next {}", batch_pos, next_token);

            if end_batches.contains(&batch_pos) {
                continue;
            }
            if next_token == eos_token {
                println!("continue");
                end_batches.insert(batch_pos.clone());
                continue;
            }

            if sample_pos >= (prompt_batch_tokens_lens[batch_pos] - prompt_row_tokens_len_min) && sample_pos < (sample_len + (prompt_batch_tokens_lens[batch_pos] - prompt_row_tokens_len_min)) {
                if let Some(t) = tokenizers[batch_pos].next_token(next_token)? {
                    if batch_pos < result_batch_tokens.len() {
                        println!("p1, {}, {}", sample_pos, batch_pos);
                        result_batch_tokens[batch_pos].push(t);
                    } else {
                        println!("p2, {}, {}", sample_pos, batch_pos);
                        result_batch_tokens.push(vec![t]);
                    }
                } else {
                    println!("no token");
                }
            }
        }
    }

    for batch_pos in 0..batch_tokens.len() {
        if let Some(rest) = tokenizers[batch_pos]
            .decode_rest()
            .map_err(anyhow::Error::msg)?
        {
            result_batch_tokens[batch_pos].push(rest);
        }
    }
    Ok(result_batch_tokens)
}

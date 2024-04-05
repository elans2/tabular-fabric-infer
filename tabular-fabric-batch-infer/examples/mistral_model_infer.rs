use arrow::array::StringArray;
use arrow::array::UInt64Array;
use arrow::record_batch::RecordBatch;

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tabular_fabric_batch_infer::base::{InferContext, ModelInfer};
use tabular_fabric_batch_infer::models::mistral::CandleMistralModelInfer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut infer = CandleMistralModelInfer::new();

    let batch = RecordBatch::try_from_iter(vec![(
        "col",
        Arc::new(StringArray::from(vec![
            //"Here is a sample quick sort implementation in rust".to_string(),
            //"Here is a sample quick sort implementation in rust".to_string(),
            "what about mistral model ?".to_string(),
        ])) as _,
    )])
    .unwrap();

    let infer_context = InferContext::default();

    let mut load_options = HashMap::new();

    let model_repo = env::var("MODEL_REPOS").expect("model repo is not set");

    load_options.insert(
        "tokenizer_file".to_string(),
        format!("{}/Mistral-7B-Instruct-v0.2/tokenizer.json", model_repo),
    );
    load_options.insert(
        "weight_files".to_string(),
        format!(
            "{}/Mistral-7B-Instruct-v0.2/model-00001-of-00003.safetensors,{}/Mistral-7B-Instruct-v0.2/model-00002-of-00003.safetensors,{}/Mistral-7B-Instruct-v0.2/model-00003-of-00003.safetensors",
            model_repo, model_repo, model_repo
        ),
    );

    load_options.insert("max_batch_size".to_string(), "2".to_string());

    println!("{:#?}", load_options);

    infer.load(load_options);
    println!("loaded model");

    let timer = Instant::now();

    let mut infer_options = HashMap::new();
    infer_options.insert("sample_len".to_string(), "2".to_string());
    infer.infer(&batch, &infer_context, infer_options);

    println!("{:#?}", timer.elapsed());

    Ok(())
}

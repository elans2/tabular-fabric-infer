use arrow::array::StringArray;
use arrow::array::UInt64Array;
use arrow::record_batch::RecordBatch;

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tabular_fabric_batch_infer::base::{InferContext, ModelInfer};
use tabular_fabric_batch_infer::models::ggml::GgmlLLamaModelInfer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut infer = GgmlLLamaModelInfer::new();

    let batch = RecordBatch::try_from_iter(vec![(
        "col",
        Arc::new(StringArray::from(vec![
            //"Here is a sample quick sort implementation in rust".to_string(),
            //"Here is a sample quick sort implementation in rust".to_string(),
            "what about phi-2 model ?".to_string(),
            "what about phi-2 model ?".to_string(),
        ])) as _,
    )])
    .unwrap();

    let infer_context = InferContext::default();

    let mut load_options = HashMap::new();

    let model_repo = env::var("MODEL_REPOS").expect("model repo is not set");

    load_options.insert(
        "model_file".to_string(),
        format!("{}/phi-2/phi-2-Q5_K_M.gguf", model_repo),
    );

    infer.load(load_options);
    let result = infer.infer(&batch, &infer_context, HashMap::new()).unwrap();

    println!("{:#?}", result);

    Ok(())
}

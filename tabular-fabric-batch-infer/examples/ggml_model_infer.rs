use arrow::array::StringArray;
use arrow::array::UInt64Array;
use arrow::record_batch::RecordBatch;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tabular_fabric_batch_infer::base::{InferContext, ModelInfer};
use tabular_fabric_batch_infer::models::ggml::GgmlLLamaModelInfer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let infer = GgmlLLamaModelInfer::new();

    let batch = RecordBatch::try_from_iter(vec![(
        "col",
        Arc::new(StringArray::from(vec![
            //"Here is a sample quick sort implementation in rust".to_string(),
            //"Here is a sample quick sort implementation in rust".to_string(),
            "<sentence>TAKE BACK RETURN</sentence>, split sentence, take first word".to_string(),
        ])) as _,
    )])
    .unwrap();

    let infer_context = InferContext::default();

    let mut load_options = HashMap::new();
    load_options.insert(
        "model_file".to_string(),
        "/Users/elans2/workspace/light/models/Phi1.5-1.8B/model.safetensors".to_string(),
    );

    infer.load(load_options);
    infer.infer(&batch, &infer_context, HashMap::new());

    Ok(())
}

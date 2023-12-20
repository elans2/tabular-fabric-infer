use crate::errors::InferenceError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

pub trait GeneralInnerModelInfer {
    fn load(
        &self,
        sources: HashMap<String, String>,
        options: HashMap<String, String>,
    ) -> Result<bool, InferenceError>;

    fn infer(
        &self,
        batch: &RecordBatch,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferenceError>;
}

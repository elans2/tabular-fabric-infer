use crate::errors::InferenceError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

pub struct InferContext {
    pub share_slices: HashMap<String, String>,
}

pub trait GeneralInnerModelInfer {
    fn load(
        &self,
        sources: HashMap<String, String>,
        options: HashMap<String, String>,
    ) -> Result<bool, InferenceError>;

    fn infer(
        &self,
        batch: &RecordBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferenceError>;
}

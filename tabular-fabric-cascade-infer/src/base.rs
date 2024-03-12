use crate::errors::InferError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

pub struct InferContext {
    pub share_slices: HashMap<String, String>,
}

impl Default for InferContext {
    fn default() -> Self {
        InferContext {
            share_slices: Default::default(),
        }
    }
}

pub trait GeneralInnerModelInfer {
    fn load(
        &self,
        options: HashMap<String, String>,
    ) -> Result<bool, InferError>;

    fn infer(
        &self,
        batch: &RecordBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<RecordBatch, InferError>;
}

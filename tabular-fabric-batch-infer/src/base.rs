use crate::errors::InferError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

pub struct InferContext {
    pub share_store: HashMap<String, String>,
}

impl Default for InferContext {
    fn default() -> Self {
        InferContext {
            share_store: Default::default(),
        }
    }
}

pub trait ModelInfer {

    fn file_resources(&self) -> Vec<String>;

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

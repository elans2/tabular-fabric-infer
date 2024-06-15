use crate::errors::InferError;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct InferBatch {
    column_names: Vec<String>,
    column_values: HashMap<String, Vec<String>>,
    num_rows: usize,
}

impl InferBatch {
    pub fn new(
        column_names: &Vec<String>,
        column_values: &HashMap<String, Vec<String>>,
    ) -> Result<Self, InferError> {
        let num_rows = column_values.values().next().map_or(0, |v| v.len());
        if column_values.values().any(|v| v.len() != num_rows) {
            return Err(InferError::GenericError {
                msg: "not equal num cells".to_string(),
            });
        }
        Ok(InferBatch {
            column_names: column_names.clone(),
            column_values: column_values.clone(),
            num_rows,
        })
    }

    pub fn column_names(&self) -> &Vec<String> {
        &self.column_names
    }

    pub fn column_values(&self, column_name: &str) -> Option<&Vec<String>> {
        self.column_values.get(column_name)
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
}

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

    fn load(&mut self, options: HashMap<String, String>) -> Result<bool, InferError>;

    fn infer(
        &mut self,
        batch: &InferBatch,
        context: &InferContext,
        options: HashMap<String, String>,
    ) -> Result<InferBatch, InferError>;
}

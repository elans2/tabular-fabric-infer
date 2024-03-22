use arrow::error::ArrowError;
use std::backtrace::Backtrace;
use candle_core;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferError {
    #[error("generic error: {msg}")]
    GenericError { msg: String },

    #[error("candle error: {source}")]
    CandleError {
        source: candle_core::Error,
    },

    #[error("arrow error: {source}")]
    ArrowError {
        source: ArrowError,
    },

    #[error("unspecified inference error: {msg}, {source}")]
    UnspecifiedError {
        msg: String,
        source: anyhow::Error,
    },
}


impl From<String> for InferError {
    fn from(msg: String) -> Self {
        InferError::GenericError {
            msg,
            //backtrace: Backtrace::capture(),
        }
    }
}

impl From<anyhow::Error> for InferError {
    fn from(err: anyhow::Error) -> Self {
        InferError::UnspecifiedError {
            msg: err.to_string(),
            source: err,
        }
    }
}

impl From<candle_core::Error> for InferError {
    fn from(err: candle_core::Error) -> Self {
        InferError::CandleError {
            source: err,
        }
    }
}

impl From<ArrowError> for InferError {
    fn from(err: ArrowError) -> Self {
        InferError::ArrowError {
            source: err,
        }
    }
}

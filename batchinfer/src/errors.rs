use arrow::error::ArrowError;
use std::backtrace::Backtrace;
use candle_core;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
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


impl From<String> for InferenceError {
    fn from(msg: String) -> Self {
        InferenceError::GenericError {
            msg,
            //backtrace: Backtrace::capture(),
        }
    }
}

impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        InferenceError::UnspecifiedError {
            msg: err.to_string(),
            source: err,
        }
    }
}

impl From<candle_core::Error> for InferenceError {
    fn from(err: candle_core::Error) -> Self {
        InferenceError::CandleError {
            source: err,
        }
    }
}

impl From<ArrowError> for InferenceError {
    fn from(err: ArrowError) -> Self {
        InferenceError::ArrowError {
            source: err,
        }
    }
}

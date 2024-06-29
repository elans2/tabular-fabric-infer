mod trace;

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tide::prelude::*; // Pulls in the json! macro.
use tide::{Body, Request};
use tabular_fabric_batch_infer::base::{InferBatch, InferContext, ModelInfer};
use tabular_fabric_batch_infer::infer::models::llama::CandleLlamaModelInfer;
use dotenv_config::EnvConfig;
use dotenvy::dotenv;
use tracing::info;
use tabular_fabric_batch_infer::errors::InferError;
use tabular_fabric_batch_infer::infer::models::mistral::CandleMistralModelInfer;
use crate::trace::ServeLayer;


#[derive(Debug, EnvConfig, Clone)]
pub struct Config {
    #[env_config(
        name = "TABULAR_INFERENCE_CORE_SERVER_HOST",
        default = "0.0.0.0"
    )]
    pub serve_host: String,

    #[env_config(
        name = "TABULAR_INFERENCE_CORE_SERVER_PORT",
        default = 50801
    )]
    pub serve_port: i32,

    #[env_config(
        name = "TABULAR_INFERENCE_CORE_MODEL_CONFIG",
        default = "modelfile.json"
    )]
    pub model_file: String,

    #[env_config(
        name = "TABULAR_INFERENCE_CORE_TIMEZONE",
        default = "Asia/Shanghai"
    )]
    pub timezone: String,

    #[env_config(name = "TABULAR_LOG_FORMAT", default = "text")]
    pub log_format: String,

    #[env_config(name = "TABULAR_LOG_LEVEL", default = "INFO")]
    pub log_level: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct InferPlainRequest {
    prompts: Vec<String>,
    options: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct InferPlainResponse {
    gen: Vec<String>,
}


#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    model: String,
    load_options: HashMap<String, String>,
    infer_options: HashMap<String, String>,
}


fn load_model_infer(model_config: &ModelConfig) -> Result<Arc<Mutex<RefCell<Box<dyn ModelInfer + Send + Sync>>>>, InferError> {
    let mut model_infer: Box<dyn ModelInfer + Send + Sync> = match model_config.model.clone().as_str() {
        "llama" => Box::new(CandleLlamaModelInfer::new()),
        "mistral" =>Box::new(CandleMistralModelInfer::new()),
        _ => {
            return Err(InferError::GenericError { msg: format!("model type not supported: {}", model_config.model) })
        }
    };

    model_infer.load(model_config.load_options.clone()).unwrap();
    let model_infer = Arc::new(Mutex::new(RefCell::new(model_infer)));
    Ok(model_infer)
}

#[derive(Clone)]
struct State {
    pub infer: Arc<Mutex<RefCell<Box<dyn ModelInfer + Sync + Send>>>>,
    pub infer_options: HashMap<String, String>,
}

#[async_std::main]
async fn main() -> tide::Result<()> {

    if let Ok(o) = dotenv() {
        println!("config loaded from env file: {:?}", o);
    } else {
        println!("config loaded from env");
    }

    let config = Config::init().expect("Failed to load config from env");

    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        tracing_subscriber::registry()
            .with(ServeLayer {
                app: "tabular-infer-standalone-server".to_string(),
                log_type: "general".to_string(),
                log_format: config.log_format.clone(),
                tz: config.timezone.clone(),
                level: Some(config.log_level.clone()),
            })
            .init();
    }
    info!("config loaded: {:?}", config);
    info!("Starting executor with config: {:?}", config);

    println!("{}", config.model_file);
    let model_config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(config.model_file).unwrap()).unwrap();

    let model_infer = load_model_infer(&model_config).unwrap();
    let state = State {
        infer: model_infer,
        infer_options: model_config.infer_options.clone(),
    };
    let mut app = tide::with_state(state);
    app.with(tide::log::LogMiddleware::new());

    app.at("/v1/infer-plain").post(|mut req: Request<State>| async move {
        let infer_req: InferPlainRequest = req.body_json().await?;
        println!("cat name: {:#?}", infer_req);

        let batch = InferBatch::new(
            &vec!["col".to_string()],
            &HashMap::from([(
                "col".to_string(),
                infer_req.prompts.clone(),
            )]),
        ).unwrap();

        let infer_context = InferContext::default();

        let mut req_infer_options = infer_req.options.clone();
        let mut infer_options: HashMap<String, String> = req.state().infer_options.clone();
        infer_options.extend(req_infer_options);

        let infer = &req.state().infer;

        let binding = infer.try_lock().unwrap();
        let mut binding = binding.borrow_mut();
        let result_batch = binding.infer(&batch, &infer_context, infer_options.clone()).unwrap();
        let result_col = result_batch.column_names().first().unwrap().to_string();
        println!("result col, {}", result_col);
        let result_values = result_batch.column_values(result_col.as_str()).unwrap().clone();
        let infer_rep = InferPlainResponse {
            gen: result_values,
        };
        Body::from_json(&infer_rep)
    });

    let listen_addr = format!("{}:{}", config.serve_host, config.serve_port);
    app.listen(listen_addr).await?;
    Ok(())
}

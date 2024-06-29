use std::collections::BTreeMap;

use chrono::Utc;
use chrono_tz::Tz;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::Layer;

pub struct ServeLayer {
    pub app: String,
    pub log_type: String,
    pub log_format: String,
    pub tz: String,
    pub level: Option<String>,
}

impl<S> Layer<S> for ServeLayer
where
    S: tracing::Subscriber,
{
    fn max_level_hint(&self) -> Option<LevelFilter> {
        if self.level.is_some() {
            match self.level.clone().unwrap().to_lowercase().as_str() {
                "trace" => Some(LevelFilter::TRACE),
                "debug" => Some(LevelFilter::DEBUG),
                "info" => Some(LevelFilter::INFO),
                "warn" => Some(LevelFilter::WARN),
                "error" => Some(LevelFilter::ERROR),
                _ => None,
            }
        } else {
            None
        }
    }

    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = BTreeMap::new();
        let mut visitor = JsonVisitor(&mut fields);
        event.record(&mut visitor);

        let tz: Tz = self.tz.parse().unwrap();
        let ts = Utc::now()
            .with_timezone(&tz)
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();
        let msg = match fields.get("message") {
            Some(msg) => format!("{}", msg)
                .strip_suffix("\"")
                .unwrap()
                .strip_prefix("\"")
                .unwrap()
                .to_string(),
            None => "".to_string(),
        };
        fields.remove("message");

        if self.log_format.clone() == "json" {
            let output = serde_json::json!({
                "_TS_": ts,
                "_MSM_": msg,
                "_LEVEL_": format!("{}", event.metadata().level().as_str()),
                "_CALLER_": format!("{}", event.metadata().name().to_string()),
                "_FIELDS_": fields,
                "_APP_": self.app.clone(),
                "_TYPE_": self.log_type.clone(),
            });
            println!("{}", serde_json::to_string(&output).unwrap());
        } else if self.log_format.clone() == "text" {
            let output = format!(
                "{} [{}] {} -- {}",
                ts,
                event.metadata().level().as_str(),
                msg,
                event.metadata().name(),
            );
            println!("{}", &output);
        }
    }
}

struct JsonVisitor<'a>(&'a mut BTreeMap<String, serde_json::Value>);

impl<'a> tracing::field::Visit for JsonVisitor<'a> {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_error(
        &mut self,
        field: &tracing::field::Field,
        value: &(dyn std::error::Error + 'static),
    ) {
        self.0.insert(
            field.name().to_string(),
            serde_json::json!(value.to_string()),
        );
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            serde_json::json!(format!("{:?}", value)),
        );
    }
}

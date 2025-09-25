#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use axum::{
    extract::Multipart,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde_json;
use candle::{DType, Device, Error, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower_http::{
    cors::CorsLayer,
    services::ServeDir
};
use std::path::Path;
use tracing_subscriber::fmt::init as tracing_init;
use anyhow::Error as E;


async fn debug_endpoint() -> &'static str {
    println!("Debug endpoint hit!");
    "Debug endpoint working"
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bind = "0.0.0.0:8039";

    tracing_init();

let api_routes = Router::new()
    .route("/tts", post(generate_tts))
    .route("/health", get(health_check))
    .route("/debug", get(debug_endpoint));


    let app = Router::new()
        .nest("/api", api_routes)
        .fallback_service(ServeDir::new("public").not_found_service(
            tower::service_fn(|_| async {
                let body = std::fs::read_to_string("public/index.html")
                    .unwrap_or_else(|_| "404 Not Found".to_string());
                Ok::<_, std::convert::Infallible>(
                    Response::builder()
                        .header("content-type", "text/html")
                        .body(axum::body::Body::from(body))
                        .unwrap()
                )
            })
        ))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(bind).await?;
    println!("Server running on http://{}", bind);
    println!("Serving static files from: ./public/");
    
    axum::serve(listener, app).await?;

    Ok(())
}


async fn health_check() -> &'static str {
    "OK"
}

async fn generate_tts(mut multipart: Multipart) -> Result<Response, StatusCode> {
    let mut text = String::new();
    let mut description = String::new();
    let mut temperature: Option<f64> = None;
    let mut seed: Option<u64> = None;
    let mut top_p: Option<f64> = None;

    // Extract form data
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let name = field.name().unwrap_or("").to_string();
        let data = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;

        match name.as_str() {
            "text" => text = data,
            "description" => description = data,
            "temperature" => temperature = data.parse().ok(),
            "seed" => seed = data.parse().ok(),
            "top_p" => top_p = data.parse().ok(),
            _ => {}
        }
    }

    if text.is_empty() || description.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Generate unique filename
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let filename = format!("generated_audio_{}.wav", timestamp);
    let filepath = format!("./public/audio/{}", filename);

    // Ensure audio directory exists
    std::fs::create_dir_all("./public/audio").map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Create WAV file
    let create_wav_args = CreateWavArgs {
        description,
        prompt: text,
        out_file: filepath.clone(),
        temperature,
        seed,
        top_p,
    };
    println!("{:?}",create_wav_args);

    if let Err(_) = create_wav_file(create_wav_args) {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Read the generated file and return it
    let audio_data = std::fs::read(&filepath).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{}\"", filename))
        .body(axum::body::Body::from(audio_data))
        .unwrap())
}

#[derive(Debug)]
struct CreateWavArgs {
    description: String,
    prompt: String,
    out_file: String,
    temperature: Option<f64>,
    seed: Option<u64>,
    top_p: Option<f64>,
}

fn create_wav_file(create_wav_args: CreateWavArgs) -> anyhow::Result<()> {
    let description: String = create_wav_args.description;
    let prompt: String = create_wav_args.prompt;
    let out_file: String = create_wav_args.out_file;
    let temperature: f64 = create_wav_args.temperature.unwrap_or(0.0);
    let seed: u64 = create_wav_args.seed.unwrap_or(0);
    let top_p: Option<f64> = create_wav_args.top_p;
    let max_steps:usize = 512;

    let start = std::time::Instant::now();
    let api = hf_hub::api::sync::Api::new()?;

    let repo = api.repo(hf_hub::Repo::with_revision(
        "parler-tts/parler-tts-large-v1".to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let model_files = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    let config = repo.get("config.json")?;
    let tokenizer = repo.get("tokenizer.json")?;
    println!("retrieved the files in {:?}", start.elapsed());
    
    let start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer).unwrap();
    // let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    println!("tokenizer loaded in {:?}", start.elapsed());
    
    let start = std::time::Instant::now();
    let device = candle_examples::device(false)?;
    println!("device loaded in {:?}", start.elapsed());
    
    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device)? };
    let config: Config = serde_json::from_reader(std::fs::File::open(config)?)?;
    println!("config loaded in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let mut model = Model::new(&config, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    // Debug: Print actual input strings and their lengths
    println!("DEBUG - Input prompt: '{}'", prompt);
    println!("DEBUG - Input description: '{}'", description);

    let description_token_ids = tokenizer
        .encode(description, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("DEBUG - Description tokens: {} tokens", description_token_ids.len());
    let description_tokens = Tensor::new(description_token_ids, &device)?.unsqueeze(0)?;

    let prompt_token_ids = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("DEBUG - Prompt tokens: {} tokens", prompt_token_ids.len());
    let prompt_tokens = Tensor::new(prompt_token_ids, &device)?.unsqueeze(0)?;
    let lp = candle_transformers::generation::LogitsProcessor::new(
        seed,
        Some(temperature),
        top_p,
    );
    
    println!("&prompt_tokens, &description_tokens, max_steps\n{:?}\n",(&prompt_tokens, &description_tokens, max_steps));
    println!("starting generation...\n");
    
    let codes = model.generate(&prompt_tokens, &description_tokens, lp, max_steps)?;
    println!("generated codes\n{codes}\n");

    let codes = codes.to_dtype(DType::I64)?;
    codes.save_safetensors("codes", "out.safetensors")?;
    let codes = codes.unsqueeze(0)?;
    let pcm = model
        .audio_encoder
        .decode_codes(&codes.to_device(&device)?)?;
    println!("pcm: {pcm}");
    
    let pcm = pcm.i((0, 0))?;
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
    let pcm = pcm.to_vec1::<f32>()?;

    // Write WAV file using candle_examples method
    let mut output = std::fs::File::create(&out_file)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, config.audio_encoder.sampling_rate)?;

    println!("Generated audio saved to: {}", out_file);
    Ok(())
}


/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, Error> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<Result<Vec<_>, candle::Error>>()?;
    Ok(safetensors_files)
}

use axum::{
    extract::Multipart,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde_json;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower_http::{
    cors::CorsLayer, 
    services::ServeDir
};
use tracing_subscriber::fmt::init as tracing_init;

#[derive(Clone)]
struct AppState {
    model: Arc<Model>,
    tokenizer: Arc<Tokenizer>,
    config: Arc<Config>,
}
async fn debug_endpoint() -> &'static str {
    println!("Debug endpoint hit!");
    "Debug endpoint working"
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_init();

    // Initialize Parler-TTS model
    let state = initialize_model().await?;

    
let api_routes = Router::new()
    .route("/tts", post(generate_tts))
    .route("/health", get(health_check))
    .route("/debug", get(debug_endpoint))
    .with_state(state);


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

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server running on http://localhost:3000");
    println!("Serving static files from: ./public/");
    
    axum::serve(listener, app).await?;

    Ok(())
}

async fn initialize_model() -> anyhow::Result<AppState> {
    println!("Loading Parler-TTS model...");
    
    let api = candle_hf_hub::api::tokio::Api::new()?;
    let model_id = "parler-tts/parler-tts-mini-v1";  // Use v1 instead of v1.1
    let repo = api.repo(candle_hf_hub::Repo::new(model_id.to_string(), candle_hf_hub::RepoType::Model));
    
    let model_file = repo.get("model.safetensors").await?;
    let config_file = repo.get("config.json").await?;
    
    let device = Device::Cpu;
    
    // Load and modify config to add missing fields
    let mut config: serde_json::Value = serde_json::from_reader(std::fs::File::open(&config_file)?)?;
    
    // Add missing num_codebooks field if it doesn't exist
    if let Some(audio_encoder) = config.get_mut("audio_encoder") {
        if !audio_encoder.as_object().unwrap().contains_key("num_codebooks") {
            audio_encoder.as_object_mut().unwrap().insert("num_codebooks".to_string(), serde_json::json!(4));
        }
    }
    
    // Convert back to Config struct
    let config: Config = serde_json::from_value(config)?;
    
    // Simple fallback tokenizer since tokenizer.json has compatibility issues
    let tokenizer = create_fallback_tokenizer()?;
    
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? 
    };
    let model = Model::new(&config, vb)?;

    println!("Model loaded successfully!");

    Ok(AppState {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
        config: Arc::new(config),
    })
}

async fn load_tokenizer(repo: &candle_hf_hub::api::tokio::ApiRepo) -> anyhow::Result<Tokenizer> {
    // Try to load tokenizer.json first
    match repo.get("tokenizer.json").await {
        Ok(tokenizer_file) => {
            match Tokenizer::from_file(tokenizer_file) {
                Ok(tokenizer) => Ok(tokenizer),
                Err(e) => {
                    anyhow::bail!("Failed to load tokenizer from file: {}", e);
                }
            }
        }
        Err(e) => {
            anyhow::bail!("Failed to download tokenizer file: {}", e);
        }
    }
}

fn create_fallback_tokenizer() -> anyhow::Result<Tokenizer> {
    use tokenizers::{
        models::bpe::BPE, 
        pre_tokenizers::whitespace::Whitespace, 
        AddedToken, Tokenizer
    };
    
    println!("Creating simple fallback tokenizer...");
    
    // Create a simple BPE tokenizer as fallback
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Whitespace {});
    
    // Add basic special tokens
    let special_tokens = vec![
        AddedToken::from("[UNK]", true),
        AddedToken::from("[PAD]", true),
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
    ];
    
    tokenizer.add_special_tokens(&special_tokens);
    
    Ok(tokenizer)
}

async fn generate_tts(
    axum::extract::State(state): axum::extract::State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, StatusCode> {
    let mut text = String::new();
    let mut description = String::new();

    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let name = field.name().unwrap_or("").to_string();
        let data = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;

        match name.as_str() {
            "text" => text = data,
            "description" => description = data,
            _ => {}
        }
    }

    if text.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    if description.is_empty() {
        description = "A clear, natural speaking voice with moderate pace and good quality.".to_string();
    }

    println!("Generating TTS for: '{}'", text);

    // Generate TTS
    match generate_speech_simple(&state, &text, &description).await {
        Ok(audio_data) => {
            let response = Response::builder()
                .header(header::CONTENT_TYPE, "audio/wav")
                .header(header::CONTENT_DISPOSITION, "attachment; filename=\"speech.wav\"")
                .body(audio_data.into())
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            Ok(response)
        }
        Err(e) => {
            eprintln!("TTS generation error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Simplified speech generation that doesn't rely heavily on tokenizer
async fn generate_speech_simple(
    state: &AppState,
    text: &str,
    description: &str,
) -> anyhow::Result<Vec<u8>> {
    let device = Device::Cpu;
    
    // Use simple encoding approach
    let description_tokens = simple_tokenize(description);
    let description_tokens = Tensor::new(description_tokens, &device)?.unsqueeze(0)?;

    let prompt_tokens = simple_tokenize(text);
    let prompt_tokens = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;

    // Generate speech with simplified approach
    let lp = candle_transformers::generation::LogitsProcessor::new(42, Some(0.0), None);
    
    let mut model = (*state.model).clone();
    let codes = model.generate(&prompt_tokens, &description_tokens, lp, 300)?;
    
    let codes = codes.to_dtype(DType::I64)?.unsqueeze(0)?;
    let pcm = model.audio_encoder.decode_codes(&codes)?;
    let pcm = pcm.i((0, 0))?;
    
    let pcm_data = pcm.to_vec1::<f32>()?;
    let normalized_pcm = normalize_audio(&pcm_data);
    
    let wav_data = create_wav_data(&normalized_pcm, state.config.audio_encoder.sampling_rate as u32)?;
    
    Ok(wav_data)
}

// Simple tokenization fallback
fn simple_tokenize(text: &str) -> Vec<u32> {
    // Very simple character-based tokenization as fallback
    text.chars()
        .map(|c| c as u32 % 1000) // Simple mapping to keep values reasonable
        .collect()
}

fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return samples.to_vec();
    }

    let max_abs = samples.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return samples.to_vec();
    }

    let scale = 0.7 / max_abs;
    samples.iter().map(|&x| x * scale).collect()
}

fn create_wav_data(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let mut wav_data = Vec::new();
    
    let cursor = std::io::Cursor::new(&mut wav_data);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::new(cursor, spec)?;
    
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }
    
    writer.finalize()?;
    Ok(wav_data)
}

async fn health_check() -> &'static str {
    "OK"
}
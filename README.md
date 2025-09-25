# TTS Server (ttser)

A Text-to-Speech (TTS) application using Rust and WASM with Parler-TTS model integration.

If you are looking for a pure rust **pure rust** TTS system that is **production ready**, you are in the wrong place. 😉

![Screenshot](<screenshot.png>)

## Architecture

This project consists of:

- **Backend**: Axum-based Rust server that uses Parler-TTS model for speech generation
- **Frontend**: WASM-compiled Rust library for browser audio functionality
- **Static Files**: HTML frontend served from the backend's public directory

## Directory Structure

```
├── backend/          # Axum HTTP server with TTS API endpoints
├── frontend/         # WASM module compiled from Rust for browser audio functionality
├── scripts/          # Build and development scripts
└── public/           # Static frontend files served by the backend
```

## Features

- Text-to-speech generation using Parler-TTS large model
- Customizable voice descriptions
- Adjustable generation parameters:
  - **Temperature**: Controls randomness (0.0-2.0)
  - **Seed**: Random seed for reproducible generation
  - **Top P**: Nucleus sampling parameter (0.0-1.0)
- Hardware acceleration support (CUDA, Metal, MKL, Accelerate)
- Web-based interface for easy interaction

## Development Setup

### Prerequisites

- Rust (latest stable)
- wasm-pack
- Node.js (for frontend dependencies)

### Building

1. **Build WASM frontend:**
   ```bash
   cd scripts && ./build.sh
   ```

2. **Build backend:**
   ```bash
   cd backend && cargo check && cargo build
   ```

3. **Build for release:**
   ```bash
   cd backend && cargo build --release
   ```

### Hardware Acceleration

The backend automatically detects and uses the best available acceleration:

> This project is only tested with CUDA

```bash
# Build with CUDA support (NVIDIA GPUs)
cd backend && cargo build --release --features cuda

# Build with Metal support (Apple Silicon)
cd backend && cargo build --release --features metal

# Build with MKL support (Intel CPU optimization)
cd backend && cargo build --release --features mkl

# Build with Accelerate support (macOS CPU optimization)
cd backend && cargo build --release --features accelerate
```

### Running

**Start development server:**
```bash
cd scripts && ./start.sh
```

**Or manually:**
```bash
cd scripts && ./build.sh && cd ../backend && cargo run
```

The server runs on `http://localhost:8039` (or configured port) and serves:
- API endpoints under `/api/*`
- Static frontend files from `/backend/public/`

### Command Line Options

- `--cpu`: Force CPU usage instead of GPU acceleration
- `--bind <ADDRESS>`: Set bind address (default: 0.0.0.0:8039)

## API Endpoints

- `POST /api/tts` - Generate speech from text
  - Form parameters:
    - `text`: Text to convert to speech
    - `description`: Voice description
    - `temperature`: Generation temperature (optional)
    - `seed`: Random seed (optional)
    - `top_p`: Top-p sampling parameter (optional)
- `GET /api/health` - Health check
- `GET /api/debug` - Debug endpoint

## Usage

1. Open your browser to `http://localhost:8039`
2. Enter text to convert to speech
3. Provide a voice description (e.g., "A female speaker with clear, animated speech")
4. Adjust generation parameters as needed:
   - **Temperature**: Higher values (1.0+) for more creative/varied output, lower values (0.0-0.5) for more consistent output
   - **Seed**: Set to a specific number for reproducible results
   - **Top P**: Controls diversity of word choice (0.9 is a good default)
5. Click "Generate Speech" to create and play audio

## Dependencies

### Backend
- **axum** - HTTP web framework
- **candle** - ML framework for running Parler-TTS model
- **tokio** - Async runtime
- **tower-http** - HTTP middleware (CORS, static files)

### Frontend
- **wasm-bindgen** - Rust/WASM/JS interop
- **web-sys** - Web API bindings for audio recording
- **js-sys** - JavaScript type bindings

## Hardware Requirements

- **CPU**: Any modern CPU (Intel/AMD/ARM)
- **GPU** (optional): NVIDIA GPU with CUDA support or Apple Silicon for acceleration
- **RAM**: Minimum 4GB, 8GB+ recommended for better performance
- **Storage**: ~2GB for model files (downloaded automatically)

## License

MIT or APACHE
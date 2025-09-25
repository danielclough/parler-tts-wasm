#!/bin/bash
set -e

echo "Building WASM frontend..."
cd frontend

# Clean previous build
cargo clean

# Build WASM package and output to public directory
wasm-pack build --target web --out-dir ./public/pkg --release

echo "WASM build complete!"
echo "Files generated in public/pkg/"

echo "Start server with: \`cd backend && cargo run --release\` or \`scripts/start.sh\`"
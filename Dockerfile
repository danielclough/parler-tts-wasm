FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM frontend
RUN chmod +x build.sh && ./build.sh

# Build backend
WORKDIR /app/tts-axum-server
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/tts-axum-server/target/release/tts-axum-server /usr/local/bin/
COPY --from=builder /app/tts-axum-server/static /static

WORKDIR /
EXPOSE 3000
CMD ["tts-axum-server"]

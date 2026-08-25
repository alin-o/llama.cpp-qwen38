# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.3.1-devel-ubuntu26.04 AS build

RUN apt-get update && apt-get install -y \
    cmake build-essential git libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN --mount=type=cache,id=qwen38-cmake-cuda133-sm89,target=/workspace/build,sharing=locked \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_STATIC=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=1 \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \        
    -DGGML_CUDA_PEER_MAX_BATCH=4096 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
    && cmake --build build --config Release --target llama-server llama-cli llama-bench llama-perplexity test-chat test-arg-parser test-backend-ops -j$(nproc) \
    && rm -rf /out \
    && mkdir -p /out \
    && cp -a build/bin/. /out/

FROM nvidia/cuda:13.3.1-runtime-ubuntu26.04

RUN apt-get update && apt-get install -y \
    curl libssl3t64 libcurl4t64 python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /usr/local/bin/

EXPOSE 8080
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

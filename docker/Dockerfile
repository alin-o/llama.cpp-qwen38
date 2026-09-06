# syntax=docker/dockerfile:1

# Reuse Carapa's local CUDA image for both compilation and runtime. Although a
# standalone runtime base would be smaller in isolation, it would introduce a
# second set of CUDA layers on this host.
FROM carapa-llama-cpp AS cuda-toolchain

FROM cuda-toolchain AS build

USER root

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
    && cmake --build build --config Release --target llama-server llama-cli llama-bench llama-paged llama-perplexity test-chat test-arg-parser test-backend-ops -j$(nproc) \
    && rm -rf /out \
    && mkdir -p /out \
    && cp -a build/bin/. /out/

FROM cuda-toolchain AS production

USER root
COPY --from=build /out/ /usr/local/bin/

# Carapa is also used for GPU-free image builds and carries
# NVIDIA_VISIBLE_DEVICES=void. Override that inherited build-time default in
# the runnable qwen image; otherwise nvidia-smi can see the device while CUDA
# applications are told that no devices are available.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=0

# The Carapa development image checks for its agent daemon. This image runs
# llama-server instead; Compose provides the server healthcheck.
HEALTHCHECK NONE
EXPOSE 8080
# The previous nvidia/cuda runtime stage ran as root. Carapa's `ubuntu`
# account does not necessarily have permission to open the host-provided
# /dev/nvidia* device nodes, which makes CUDA report "no GPU device found".
USER root
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

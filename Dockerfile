FROM python:3.11-slim

# torch CPU 运行时需要 OpenMP 库（python:3.11-slim 默认不带）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 单独装 CPU 版（上海交大 pytorch-wheels 镜像）：PyPI 的 torch 在 Linux 上会拉
# cuda-toolkit / nvidia-* 等 CUDA 依赖（无 GPU 用不上，白白 +2~3G），CPU 源不含这些。
RUN pip install --no-cache-dir torch \
    --index-url https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cpu/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 其余依赖走清华 PyPI（torch 已装 CPU 版，sentence-transformers 不会再拉 CUDA）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 可选：把 BGE 模型烘焙进镜像（docker build --build-arg BAKE_MODELS=1）。
# 镜像 +6.5G，换来「新机器 pull 即用、无需联网下载模型」的离线一键部署。
# 默认关闭：端到端验证 / 开发用宿主机模型缓存挂载（见 docker-compose.yml），不重复下载。
ARG BAKE_MODELS=0
RUN if [ "$BAKE_MODELS" = "1" ]; then \
        pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope && \
        python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3'); snapshot_download('BAAI/bge-reranker-v2-m3')"; \
    fi

COPY app ./app
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 基础环境：Python 3.11.16，底层系统为 Debian Bookworm
FROM python:3.11.16-slim-bookworm

# 让 Python 的日志及时显示
ENV PYTHONUNBUFFERED=1

# 安装图像处理需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装支持 CUDA 12.8 的 PyTorch 和配套 torchvision
RUN python -m pip install --no-cache-dir \
    torch==2.10.0 \
    torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 安装与你原来记录一致的 Ultralytics 版本
RUN python -m pip install --no-cache-dir \
    ultralytics==8.4.147

# 补齐 OpenCV / Qt 显示窗口需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# 检查 Python 依赖是否存在版本冲突
RUN python -m pip check

# 设置容器内默认工作目录
WORKDIR /workspace

# 默认启动 Bash 命令行
CMD ["bash"]

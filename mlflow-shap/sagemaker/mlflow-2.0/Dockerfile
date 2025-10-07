FROM public.ecr.aws/docker/library/python:3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY requirements.txt /app/

# Install deps from requirements
RUN uv pip install --system -r requirements.txt

# Force compatible training libs
RUN uv pip install --system "sagemaker-training<5.0" "smdebug==1.0.34" "protobuf<5.0"

# SageMaker env
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8
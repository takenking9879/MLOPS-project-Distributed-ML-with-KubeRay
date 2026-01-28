FROM rayproject/ray:2.9.0-py311

USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        gcc \
        g++ \
        build-essential \
        libssl-dev \
        libsasl2-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Manually install librdkafka with SASL/SSL support
RUN curl -fsSL https://github.com/confluentinc/librdkafka/archive/refs/tags/v2.12.0.tar.gz -o librdkafka.tar.gz && \
    tar -xzf librdkafka.tar.gz && \
    cd librdkafka-2.12.0 && \
    ./configure --enable-ssl --enable-sasl && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf librdkafka-2.12.0 librdkafka.tar.gz

# Copy requirements and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install uv==0.9.24 \
    && uv pip install --system --no-cache -r /app/requirements.txt

# Switch back to ray user for security
USER ray
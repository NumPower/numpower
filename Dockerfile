# Use a PHP 8 base image
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 as builder
ENV DEBIAN_FRONTEND=noninteractive
# Install CUDA and cuBLAS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libssl-dev \
    libxml2-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libonig-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libpng-dev \
    libzip-dev \
    libwebp-dev \
    curl \
    libxpm-dev \
    libfreetype6-dev \
    gnupg

# Add the CUDA repository key
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | gpg --dearmor > /usr/share/keyrings/cuda-archive-keyring.gpg

# Add the CUDA repository
RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# Install CUDA
RUN apt-get update && apt-get install -y cuda

# Install cuBLAS
RUN apt-get update && apt-get install -y libcublas-11-4

# Install required build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libssl-dev \
    libxml2-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libonig-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libpng-dev \
    libzip-dev \
    libwebp-dev \
    libxpm-dev \
    libfreetype6-dev

# Download and extract PHP source code
WORKDIR /usr/src/php
RUN apt-get install -y wget && \
    wget -O php.tar.gz https://www.php.net/distributions/php-8.2.0.tar.gz && \
    tar -xf php.tar.gz --strip-components=1 && \
    rm php.tar.gz

# Configure and compile PHP with desired options
RUN ./configure \
    --prefix=/usr/local/php \
    --with-config-file-path=/usr/local/php/etc \
    --enable-debug \
    --enable-mbstring \
    --with-curl \
    --with-openssl \
    --with-libxml \
    --with-pdo-mysql \
    --with-pdo-sqlite \
    --with-sqlite3 \
    --with-zlib \
    --with-bz2 \
    --with-jpeg \
    --with-webp \
    --with-xpm \
    --with-freetype \
    --with-zip \
    && make -j$(nproc) \
    && make install

# Base image for the final PHP runtime
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04 as runtime
ENV DEBIAN_FRONTEND=noninteractive
# Copy the compiled PHP installation from the builder stage
COPY --from=builder /usr/local/php /usr/local/php

# Set the environment variables
ENV PATH="/usr/local/php/bin:${PATH}"
ENV PHP_INI_SCAN_DIR="/usr/local/php/etc/conf.d"

# Set the environment variables for CUDA
ENV PATH="/usr/local/php/bin:${PATH}"
ENV PHP_INI_SCAN_DIR="/usr/local/php/etc/conf.d"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install necessary CUDA and cuBLAS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libssl-dev \
    libxml2-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libonig-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libpng-dev \
    libzip-dev \
    libwebp-dev \
    libxpm-dev \
    libfreetype6-dev \
    libcublas-11-4

RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libssl-dev \
    libxml2-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libonig-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libpng-dev \
    libzip-dev \
    libwebp-dev \
    libxpm-dev \
    libfreetype6-dev

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    libopenblas-dev

# Copy the PHP extension source code to the container
COPY . /src

# Set the working directory
WORKDIR /src


# Start the PHP CLI
CMD ["/bin/bash"]
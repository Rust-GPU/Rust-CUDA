FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    curl \
    libssl-dev \
    libtinfo-dev \
    pkg-config \
    xz-utils \
    zlib1g-dev

# Get LLVM 7 & libffi7
WORKDIR /data/llvm7
RUN curl -sSf -L -O http://security.ubuntu.com/ubuntu/pool/universe/libf/libffi7/libffi7_3.3-5ubuntu1_amd64.deb
RUN curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7_7.0.1-12_amd64.deb
RUN curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-dev_7.0.1-12_amd64.deb
RUN curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/libllvm7_7.0.1-12_amd64.deb
RUN curl -sSf -L -O http://mirrors.kernel.org/ubuntu/pool/universe/l/llvm-toolchain-7/llvm-7-runtime_7.0.1-12_amd64.deb
RUN apt-get install -y ./*.deb
RUN ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config

# Get Rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
ADD . /data/Rust-CUDA
WORKDIR /data/Rust-CUDA
RUN rustup show

ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
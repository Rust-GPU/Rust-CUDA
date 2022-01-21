FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y


# get prebuilt llvm
RUN curl -O https://releases.llvm.org/7.0.1/clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz &&\
    xz -d /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz &&\
    tar xf /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar &&\
    rm /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04.tar &&\
    mv /clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-18.04 /root/llvm

# set env
ENV LLVM_CONFIG=/root/llvm/bin/llvm-config
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=$CUDA_ROOT
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
ENV PATH=$CUDA_ROOT/nvvm/lib64:/root/.cargo/bin:$PATH

# make ld aware of necessary *.so libraries
RUN echo $CUDA_ROOT/lib64 >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/compat >> /etc/ld.so.conf &&\
    echo $CUDA_ROOT/nvvm/lib64 >> /etc/ld.so.conf &&\
    ldconfig

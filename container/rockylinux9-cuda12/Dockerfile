FROM nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

RUN dnf -y install \
    openssl-devel \
    pkgconfig \
    redhat-rpm-config \
    which \
    xz \
    zlib-devel

# Get LLVM 7 & libffi.so.6
WORKDIR /data/llvm7
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/9/Everything/x86_64/Packages/l/libffi3.1-3.1-36.el9.x86_64.rpm
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-7.0.1-7.el8.x86_64.rpm
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-devel-7.0.1-7.el8.x86_64.rpm
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-libs-7.0.1-7.el8.x86_64.rpm
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-static-7.0.1-7.el8.x86_64.rpm
RUN dnf -y install ./*.rpm
RUN ln -s /usr/bin/llvm-config-7-64 /usr/bin/llvm-config

# Get Rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
ADD . /data/Rust-CUDA
WORKDIR /data/Rust-CUDA
RUN rustup show

ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
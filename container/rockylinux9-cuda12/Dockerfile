FROM nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

RUN dnf -y install \
    clang \
    openssl-devel \
    pkgconfig \
    redhat-rpm-config \
    which \
    xz \
    zlib-devel && \
    dnf clean all

# Needed to build `path_tracer`, `optix/ex03_window` example
RUN dnf -y install \
    cmake \
    fontconfig-devel \
    libX11-devel  \
    libXcursor-devel \
    libXi-devel \
    libXrandr-devel && \
    dnf clean all

# Get LLVM 7 & libffi.so.6
WORKDIR /data/llvm7
RUN curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/9/Everything/x86_64/Packages/l/libffi3.1-3.1-36.el9.x86_64.rpm && \
    curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-7.0.1-7.el8.x86_64.rpm && \
    curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-devel-7.0.1-7.el8.x86_64.rpm && \
    curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-libs-7.0.1-7.el8.x86_64.rpm && \
    curl -sSf -L -O https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/l/llvm7.0-static-7.0.1-7.el8.x86_64.rpm && \
    dnf -y install ./*.rpm && \
    ln -s /usr/bin/llvm-config-7-64 /usr/bin/llvm-config && \
    rm -rf ./*.rpm && \
    dnf clean all

# Get Rust
RUN curl -sSf -L https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup the workspace
WORKDIR /data/Rust-CUDA
RUN --mount=type=bind,source=rust-toolchain.toml,target=/data/Rust-CUDA/rust-toolchain.toml \
    rustup show

# Add nvvm to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_LINK_STATIC=1
ENV RUST_LOG=info
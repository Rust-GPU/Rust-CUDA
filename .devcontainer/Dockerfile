FROM ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:latest

# install git + llvm-19
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    git && \
    llvm-19 && \
    libzstd-dev && \
    libpolly-19-dev && \
    rm -rf /var/lib/apt/lists/*

# get rid of llvm-config-7 default and put llvm-config-19
RUN rm /usr/bin/llvm-config && \
    ln -s /usr/bin/llvm-config-19 /usr/bin/llvm-config
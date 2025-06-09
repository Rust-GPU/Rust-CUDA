#!/bin/bash
# Simple backtrace decoder with addr2line
echo "=== Backtrace Analysis ==="
echo "1. rustc_driver crash:"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x373de5f
echo "2. libc (signal handler):"
addr2line -e /lib/x86_64-linux-gnu/libc.so.6 -f -C -p 0x45330
echo "3. Your NVVM code:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe6232e
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe39322
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe39da6
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xede3e3
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xde0815
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe6320e
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xdfa368
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe007af
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe9bdd0
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xef8d61
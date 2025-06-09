#!/bin/bash
# Simple backtrace decoder with addr2line
echo "=== Backtrace Analysis ==="
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe0c0f2

addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe0c0f2
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe00d00
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xdfa318
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe6321e
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xde07c5
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xede413
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe39db6
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe39332
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe6233e
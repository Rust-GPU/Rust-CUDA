#!/bin/bash
# Simple backtrace decoder with addr2line
echo "=== Backtrace Analysis ==="
echo "1. rustc_driver crash:"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x373de5f
echo "2. libc (signal handler):"
addr2line -e /lib/x86_64-linux-gnu/libc.so.6 -f -C -p 0x45330
echo "3. Your NVVM code:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xf21197

echo "3. Your NVVM code:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p _ZN107_$LT$rustc_codegen_nvvm..builder..Builder$u20$as$u20$rustc_codegen_ssa..traits..builder..BuilderMethods$GT$4call17h409da23fd3edd6a1E+0x49e

echo "4. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xec4d1e
echo "5. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe29291
echo "6. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe22c40
echo "7. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe8c989
echo "8. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe096ba
echo "9. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xf068e5
echo "10. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe62415
echo "11. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe619b5
echo "12. Called from:"
addr2line -e /workspaces/Rust-CUDA/target/debug/deps/librustc_codegen_nvvm.so -f -C -p 0xe8bad8
echo "13. rustc_driver (codegen_and_build_linker):"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x5f38b70
echo "14. rustc_driver:"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x5f03a20
echo "15. rustc_driver (thread setup):"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x5d5a4b4
echo "16. rustc_driver:"
addr2line -e /root/.rustup/toolchains/nightly-2025-03-02-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/librustc_driver-e3b06f91230294e6.so -f -C -p 0x5d5b8ab
echo "17. libc (thread start):"
addr2line -e /lib/x86_64-linux-gnu/libc.so.6 -f -C -p 0x9caa4

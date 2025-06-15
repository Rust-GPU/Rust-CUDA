# rustc_llvm_wrapper

This is basically 99.99% from https://github.com/rust-lang/rust/tree/8c392966a013fd8a09e6b78b3c8d6e442bc278e1/compiler/rustc_llvm/llvm-wrapper with some added legacy functions convert LLVM v7 -> LLVM v19.

This to workaround the fact that `rustc_codegen_nvvm` was written on LLVM v7 but we want to look towards the future (Blackwell+) which is based on LLVM v19.

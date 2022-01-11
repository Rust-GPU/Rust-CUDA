# Tips 

This section contains some tips on what to do and what not to do using the project.

## GPU kernels

- Generally don't derive `Debug` for structs in GPU crates. The codegen currently does not do much global
DCE (dead code elimination) so debug can really slow down compile times and make the PTX gigantic. This
will get much better in the future but currently it will cause some undesirable effects.

- Don't use recursion, CUDA allows it but threads have very limited stacks (local memory) and stack overflows
yield confusing `InvalidAddress` errors. If you are getting such an error, run the executable in cuda-memcheck,
it should yield a write failure to `Local` memory at an address of about 16mb. You can also put the ptx file through
`cuobjdump` and it should yield ptxas warnings for functions without a statically known stack usage.


## Build

- If `rustc_codegen_nvvm` has to rebuild on every build, you've encountered what appears to be a cargo bug. As a 
workaround, edit `target/{release/debug}/build/rustc_codegen_nvvm-<some hash>/output` to set its edit date to now (for example using `touch` on linux, or just editing some content and then setting it back). After a few rebuilds it won't rebuild again and your compile times will become sane again.
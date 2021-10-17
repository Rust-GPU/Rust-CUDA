# PTX Generation 

This is the final and most fun part of codegen, taking our LLVM bitcode and giving it to libnvvm. 
It is in theory as simple as just giving nvvm every single bitcode module, but in practice, we do a couple 
of things before and after to reduce ptx size and speed things up.

# The NVVM API

libnvvm is a dynamically linked library which is distributed in every download of the CUDA SDK. 
If you are on windows, it should be somewhere around `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/nvvm/bin`
where `v11.3` is the version of cuda you have downloaded. On Windows it's usually called `nvvm64_40_0.dll`. If you are 
on linux it should be somewhere around `/opt/cuda/nvvm-prev/lib64/libnvvm.so`. You can see its API either in the
[API docs](https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html) or in its header file in the `include` folder.

We have our own high level bindings to it published as a crate called `nvvm`.

The libnvvm API could not be simpler, it is just a couple of functions:
- Make new program
- Add bitcode module
- Lazy add bitcode module (we will cover this soon)
- Verify program
- Compile program

The first step is always making a new program, a program is just a container for 
modules that then gets compiled. 

# Module loading

This is the most important part, we need to add our LLVM bitcode to the program, that 
should be a very simple thing that would involve no calls to random functions in the rustc
haystack, ...right? Why of course not, you didn't seriously think we would make this
straight-forward, right?

So, in theory it is very simple, just load the bitcode from the rlib and tell nvvm to load it. 
While this is easy and it works, it has its own very visible issues.

Traditionally, if you never use a function, either the compiler destroys it when using LTO, or 
the linker destroys it in its own dead code pass. The issue is that LTO is not always run,
and we do not have a linker, nvvm *is* our linker. However, nvvm does not eliminate dead functions.
I think you can guess why that is a problem, so unless we want `11mb` ptx files (yes this is actually
how big it was) we need to do something about it.

# Lazy loading

Nvvm supports a special way of loading modules called lazy loading. This is the key to 
reducing our glorious fat ptx files.

Suppose you have a dependency graph like this:

```
A -> B -> C
\        /
 +------+
```

`A` is the main crate with the kernels we are compiling, and it depends on `B` and `C`, `B` also depends on `C`.

And let's say that `A` and `B` do not use every function inside of `C`. If they do not use every function, why
are we wasting time optimizing this and putting it in the PTX?

Lazy loading solves this, lazy loading looks at the previous modules to see what functions from the module we are using,
then it only loads the functions that we are using. This saves nvvm from some useless optimization and it makes our ptx 
file smaller.

Lazy loading must be done in order or it wont work, we need to load this specific graph in this order:
- 1: Load `A` as a normal (non-lazy) module
- 2: Load `B` as a lazy module
- 3: Load `C` as a lazy module

This would be complex if we had to resolve the graph ourselves, but thankfully we can just call our dearest friend 
rustc and ask them what the dependencies are using [`CStore`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/creader/struct.CStore.html#method.crate_dependencies_in_postorder).

# libdevice

There are a couple of special modules we need to load before we are done, `libdevice` and `libintrinsics`. The first and most
important one is libdevice, libdevice is essentially a bitcode module containing hyper-optimized math intrinsics
that nvidia provides for us. You can find it as a `.bc` file in the libdevice folder inside your nvvm install location.
Every function inside of it is prefixed with `__nv_`, you can find docs for it [here](https://docs.nvidia.com/cuda/libdevice-users-guide/index.html).

We declare these intrinsics inside of `ctx_intrinsics.rs` and link to them inside cuda_std. We also use them to codegen
a lot of intrinsics inside `intrinsic.rs`, such as `sqrtf32`.

libdevice is also lazy loaded so we do not import useless intrinsics.

# libintrinsics

This is the last special module we load, it is simple, it is just a dumping ground for random wrapper functions 
we need to define that `cuda_std` or the codegen needs. You can find the llvm ir definition for it in the codegen directory
called `libintrinsics.ll`. All of its functions should be declared with the `__nvvm_` prefix.

# Compilation

Finally, we have everything loaded and we can compile our program. We do one last thing however.

Nvvm has a function for verifying our program to make sure we did not add anything nvvm does not like. We run this 
before compilation just to be safe. Although annoyingly this does not catch all errors, nvvm just segfaults sometimes which is unfortunate.

Compiling is simple, we just call nvvm's program compile function and panic if it fails, if it doesn't, we get a final PTX string. We 
can then just write that to the file that rustc wants us to put the final item in.

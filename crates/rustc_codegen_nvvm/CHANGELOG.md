# Changelog

Notable changes to this project will be documented in this file.

## Unreleased

### Dead Code Elimination 

PTX files no longer include useless functions and globals, we have switched to an alternative
method of codegen for the final steps of the codegen. We no longer lazily-load modules using dependency graphs, 
we instead merge all the modules into one then run global DCE on it before giving it to libnvvm.

This means all of the dead code is gone before it gets to the libnvvm stage, drastically lowering the size of 
the built PTX and improving codegen performance.

- Trace-level debug is compiled out for release now, decreasing the size of the codegen dll and improving compile times.

## 0.1.1 - 11/26/21

- Fix things using the `bswap` intrinsic panicking.
- (internal) Run clippy and clean things up a bit.

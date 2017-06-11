# About
A multithreaded fork of bluss' [matrixmultiply](https://github.com/bluss/matrixmultiply) crate.
General matrix multiplication for f32, f64 matrices.
Allows arbitrary row, column strided matrices.
Relies heavily on llvm to vectorise the floating point ops.

To enable specialised vector instructions for you computer compile using:
`RUSTFLAGS="-C target-cpu=native"`
and
`MATMULFLAGS="flag1, flag2, ..."`
where one flag is an architecture flag:
```
arch_generic4x4               // fallback if architecture is unknown, should use x86 sse and ARM Neon
arch_generic4x4fma            // might be useful for newer ARM Neon
arch_penryn                   // uses the extra x86_64 xmm registers
arch_sandybridge              // uses AVX
arch_haswell                  // uses AVX2
```
and the rest are optional flags:
```
ftz_daz                       // on x86 this will round denormals to zero to improve performance
```
e.g. `MATMULFLAGS="arch_sandybridge, ftz_daz"`

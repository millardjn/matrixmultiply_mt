// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! 
//! General matrix multiplication for f32, f64 matrices.
//! 
//! Allows arbitrary row, column strided matrices.
//! 
//! Uses the same microkernel algorithm as [BLIS][bl], but in a much simpler
//! and less featureful implementation.
//! See their [multithreading][mt] page for a very good diagram over how
//! the algorithm partitions the matrix (*Note:* this crate implements multithreading of BLIS Loop 3).
//! 
//! [bl]: https://github.com/flame/blis
//! 
//! [mt]: https://github.com/flame/blis/wiki/Multithreading
//!
//! ## Matrix Representation
//!
//! **matrixmultiply** supports matrices with general stride, so a matrix
//! is passed using a pointer and four integers:
//!
//! - `a: *const f32`, pointer to the first element in the matrix
//! - `m: usize`, number of rows
//! - `k: usize`, number of columns
//! - `rsa: isize`, row stride
//! - `csa: isize`, column stride
//!
//! In this example, A is a m by k matrix. `a` is a pointer to the element at
//! index *0, 0*.
//!
//! The *row stride* is the pointer offset (in number of elements) to the
//! element on the next row. It’s the distance from element *i, j* to *i + 1,
//! j*.
//!
//! The *column stride* is the pointer offset (in number of elements) to the
//! element in the next column. It’s the distance from element *i, j* to *i,
//! j + 1*.
//!
//! For example for a contiguous matrix, row major strides are *rsa=k,
//! csa=1* and column major strides are *rsa=1, csa=m*.
//!
//! Stides can be negative or even zero, but for a mutable matrix elements
//! may not alias each other.

#![cfg_attr(any(ftz_daz, prefetch), feature(link_llvm_intrinsics))]

#[cfg(prefetch)]
extern {
	/// The `llvm.prefetch` intrinsic.
	/// address(a) is the address to be prefetched
	/// rw(b) is the specifier determining if the fetch should be for a read (0) or write (1)
	/// locality(c) is a temporal locality specifier ranging from (0) - no locality, to (3) - extremely local keep in cache.
	/// The cache type(d) specifies whether the prefetch is performed on the data (1) or instruction (0) cache.
	/// The rw, locality and cache type arguments must be constant integers.
	#[link_name = "llvm.prefetch"]
	fn prefetch(a: *mut i8, b: i32, c: i32, d: i32) -> ();
}

#[cfg(not(prefetch))]
fn prefetch(_a: *mut i8, _b: i32, _c: i32, _d: i32) -> (){}


#[cfg(ftz_daz)]
extern {
	/// The `llvm.x86.sse.stmxcsr` intrinsic.
	#[link_name = "llvm.x86.sse.stmxcsr"]
	fn sse_stmxcsr(a: *mut i8) -> ();
	/// The `llvm.x86.sse.ldmxcsr` intrinsic.
	#[link_name = "llvm.x86.sse.ldmxcsr"]
	fn sse_ldmxcsr(a: *mut i8) -> ();
}

#[cfg(not(ftz_daz))]
#[allow(unused_variables)]
unsafe fn sse_stmxcsr(a: *mut i8) -> (){}
#[cfg(not(ftz_daz))]
#[allow(unused_variables)]
unsafe fn sse_ldmxcsr(a: *mut i8) -> (){}



extern crate typenum_loops;
extern crate typenum;
extern crate generic_array;
extern crate num_traits;
extern crate rawpointer;
extern crate num_cpus;
extern crate threadpool;
extern crate parking_lot;
extern crate smallvec;

#[macro_use] extern crate lazy_static;

#[macro_use] mod debugmacros;


mod generic_params;
mod generic_kernel;
mod gemm;
mod util;
mod snb_kernels;
mod hwl_kernels;

pub use gemm::sgemm;
pub use gemm::dgemm;
pub use gemm::sgemm_st;
pub use gemm::dgemm_st;

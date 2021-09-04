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

/// If 'ftz_daz' feature is not enabled this does nothing and returns 0.
/// Sets the ftz and daz bits of the mxcsr register, zeroing input and result subnormal floats.
/// Returns the current mxcsr register value, which should be restored at a later time by reset_ftz_and_daz();
unsafe fn set_ftz_and_daz() -> u32 {
	if cfg!(all(ftz_daz, target_feature = "sse2")) {
		#[cfg(target_arch = "x86_64")]
		{
			let old_mxcsr = std::arch::x86_64::_mm_getcsr();
			let new_mxcsr = old_mxcsr | 0x8000 | 0x0040; // enable Flush-to-zero and Denormal-are-zero
			std::arch::x86_64::_mm_setcsr(new_mxcsr);
			old_mxcsr
		}

		#[cfg(target_arch = "x86")]
		{
			let old_mxcsr = std::arch::x86::_mm_getcsr();
			let new_mxcsr = old_mxcsr | 0x8000 | 0x0040; // enable Flush-to-zero and Denormal-are-zero
			std::arch::x86::_mm_setcsr(new_mxcsr);
			old_mxcsr
		}
	} else {
		0
	}
}

/// If 'ftz_daz' feature is not enabled this does nothing.
/// Set the mxcsr register back to its previous value
unsafe fn reset_ftz_and_daz(old_mxcsr: u32) {
	if cfg!(all(ftz_daz, target_feature = "sse2")) {
		#[cfg(target_arch = "x86_64")]
		std::arch::x86_64::_mm_setcsr(old_mxcsr);

		#[cfg(target_arch = "x86")]
		std::arch::x86::_mm_setcsr(old_mxcsr);
	}
}

unsafe fn prefetch_read(p: *const i8) {
	if cfg!(all(prefetch, target_feature = "sse2")) {
		#[cfg(target_arch = "x86_64")]
		std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p);

		#[cfg(target_arch = "x86")]
		std::arch::x86::_mm_prefetch::<{ std::arch::x86::_MM_HINT_T0 }>(p);
	}
}

unsafe fn prefetch_write(p: *const i8) {
	if cfg!(all(prefetch, target_feature = "sse2")) {
		#[cfg(target_arch = "x86_64")]
		std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_ET0 }>(p);

		#[cfg(target_arch = "x86")]
		std::arch::x86::_mm_prefetch::<{ std::arch::x86::_MM_HINT_ET0 }>(p);
	}
}

extern crate generic_array;
extern crate num_cpus;
extern crate num_traits;
extern crate parking_lot;
extern crate rawpointer;
extern crate smallvec;
extern crate threadpool;
extern crate typenum;
extern crate typenum_loops;

#[macro_use]
extern crate lazy_static;

#[macro_use]
mod debugmacros;

mod gemm;
mod generic_kernel;
mod generic_params;
mod hwl_kernels;
mod snb_kernels;
mod util;

pub use gemm::dgemm;
pub use gemm::dgemm_st;
pub use gemm::sgemm;
pub use gemm::sgemm_st;

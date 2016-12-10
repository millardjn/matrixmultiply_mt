// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use typenum::Unsigned;
use typenum_loops::Loop;
use generic_params::*;
use num::One;
use std::cmp::min;

#[inline(always)]
/// Call the GEMM kernel with a "masked" output C.
///
/// Simply redirect the MR by NR kernel output to the passed
/// in `mask_buf`, and copy the non masked region to the real
/// C.
///
/// + rows: rows of kernel unmasked
/// + cols: cols of kernel unmasked
pub unsafe fn masked_kernel<K: KernelConfig>(k: usize,
							  alpha: K::T,
							  a: *const K::T,
							  b: *const K::T,
							  c: *mut K::T,
							  rsc: isize,
							  csc: isize,
							  rows: usize,
							  cols: usize)
{
	let mr = min(K::MR::to_usize(), rows);
	let nr = min(K::NR::to_usize(), cols);
	let one = <K::T as One>::one();
	let ab = kernel_compute::<K>(k, one, a, b);
	for j in 0..nr {
		for i in 0..mr {
			//if i < rows && j < cols {
				let cptr = c.offset(rsc * i as isize + csc * j as isize);
				*cptr = *cptr + alpha * ab[i][j];
			//}
		}
	}
}

/// matrix multiplication kernel
///
/// This does the matrix multiplication:
///
/// C ← α A B + β C
///
/// + k: length of data in a, b
/// + a, b are packed
/// + c has general strides
/// + rsc: row stride of c
/// + csc: col stride of c
/// + if beta is 0, then c does not need to be initialized
#[inline(always)]
pub unsafe fn kernel<K: KernelConfig>(k: usize,
					alpha: K::T,
					a: *const K::T,
					b: *const K::T,
					c: *mut K::T,
					rsc: isize,
					csc: isize) {

		let ab = kernel_compute::<K>(k, alpha, a, b);
	
		kernel_write::<K>(c, rsc, csc, &ab);		

}


/// Split out compute for better vectorisation
#[inline(never)]
unsafe fn kernel_compute<K: KernelConfig>(k: usize, alpha: K::T, a: *const K::T, b: *const K::T) -> GA<GA<K::T, K::NR>, K::MR>{

	// Compute matrix multiplication into ab[i][j]
	let mut ab = <GA<GA<K::T, K::NR>, K::MR>>::default();

	K::KU::partial_unroll(k, |l|{
		let a = a.offset((l*K::MR::to_usize()) as isize);
		let b = b.offset((l*K::NR::to_usize()) as isize);
		K::MR::full_unroll(|i|{
			K::NR::full_unroll(|j|{
				ab[i][j] = ab[i][j] + at::<K>(a, i) * at::<K>(b, j);
			});
		});
	});

	K::MR::full_unroll(|i|{
		K::NR::full_unroll(|j|{
			ab[i][j] = ab[i][j]*alpha;
		});	
	});	

	ab
}

/// Choose writes to C in a cache/vectorisation friendly manner if possible
#[inline(always)]
unsafe fn kernel_write<K: KernelConfig>(c: *mut K::T, rsc: isize, csc: isize, ab: & GA<GA<K::T, K::NR>, K::MR>) {

	if rsc == 1 {
		K::MR::full_unroll(|i|{
			K::NR::full_unroll(|j|{
				let v = c.offset(1 * i as isize + csc * j as isize);
				*v = *v + ab[i][j];
			});	
		});	
		
	} else if csc == 1 {
		K::MR::full_unroll(|i|{
			K::NR::full_unroll(|j|{
				let v = c.offset(rsc * i as isize + 1 * j as isize);
				*v = *v + ab[i][j];
			});	
		});	
		
	} else {
		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				let v = c.offset(rsc * i as isize + csc * j as isize);
				*v = *v + ab[i][j];
			}
		}
	}

}

#[inline(always)]
unsafe fn at<K: KernelConfig>(ptr: *const K::T, i: usize) -> K::T {
	*ptr.offset(i as isize)
}

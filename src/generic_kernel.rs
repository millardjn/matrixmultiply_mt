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
use std::cmp::min;
use num_traits::Float;
use super::prefetch;

/// Call the GEMM kernel with a "masked" output C.
///
/// Simply redirect the MR by NR kernel output to the passed
/// in `mask_buf`, and copy the non masked region to the real
/// C.
///
/// + rows: rows of kernel unmasked
/// + cols: cols of kernel unmasked
//#[inline(always)]
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
	prefetch(a as *mut i8, 0, 3, 1);
	prefetch(b as *mut i8, 0, 3, 1); // addr, read, nonlocal, data
	write_prefetch::<K>(c, rsc, csc);
	if K::TR::to_usize() == 0 {
		let ab = kernel_compute::<K>(k, alpha, a, b);
		for j in 0..nr {
			for i in 0..mr {
				let cptr = c.offset(rsc * i as isize + csc * j as isize);
				*cptr = *cptr + ab[i][j];
			}
		}
	} else {
		let ab = kernel_compute_trans::<K>(k, alpha, a, b);
		for j in 0..nr {
			for i in 0..mr {
				let cptr = c.offset(rsc * i as isize + csc * j as isize);
				*cptr = *cptr + ab[j][i];
			}
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
#[inline(never)]
pub unsafe fn kernel<K: KernelConfig>(k: usize,
					alpha: K::T,
					a: *const K::T,
					b: *const K::T,
					c: *mut K::T,
					rsc: isize,
					csc: isize) {
	prefetch(a as *mut i8, 0, 3, 1);
	prefetch(b as *mut i8, 0, 3, 1); // addr, read, nonlocal, data
	write_prefetch::<K>(c, rsc, csc);
	if K::TR::to_usize() == 0 {
		let ab = kernel_compute::<K>(k, alpha, a, b);
		kernel_write::<K>(c, rsc, csc, &ab);
	} else {
		let ab = kernel_compute_trans::<K>(k, alpha, a, b);
		kernel_write_trans::<K>(c, rsc, csc, &ab);
	}

}


/// Split out compute for better vectorisation
#[inline(always)]
unsafe fn kernel_compute<K: KernelConfig>(k: usize, alpha: K::T, a: *const K::T, b: *const K::T) -> GA<GA<K::T, K::NR>, K::MR>{

	// Compute matrix multiplication into ab[i][j]
	let mut ab = <GA<GA<K::T, K::NR>, K::MR>>::default();

	K::KU::partial_unroll(k, &mut |l, _|{
		let a = a.offset((l*K::MR::to_usize()) as isize);
		let b = b.offset((l*K::NR::to_usize()) as isize);

		K::MR::full_unroll(&mut |i|{
			K::NR::full_unroll(&mut |j|{
				if K::FMA::to_usize() > 0 {
					ab[i][j] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[i][j]);
				} else {
					ab[i][j] += at::<K::T>(a, i) * at::<K::T>(b, j);
				}
			});
		});
	});

	K::MR::full_unroll(&mut |i|{
		K::NR::full_unroll(&mut |j|{
			ab[i][j] = ab[i][j]*alpha;
		});
	});

	// for i in 0..K::MR::to_usize() {
	// 	for j in 0..K::NR::to_usize() {
	// 		ab[i][j] = ab[i][j]*alpha;
	// 	}
	// }

	ab
}


/// Split out compute for better vectorisation
#[inline(always)]
unsafe fn kernel_compute_trans<K: KernelConfig>(k: usize, alpha: K::T, a: *const K::T, b: *const K::T) -> GA<GA<K::T, K::MR>, K::NR>{

	// Compute matrix multiplication into ab[i][j]
	let mut ab = <GA<GA<K::T, K::MR>, K::NR>>::default();

	K::KU::partial_unroll(k, &mut |l, _|{
		let a = a.offset((l*K::MR::to_usize()) as isize);
		let b = b.offset((l*K::NR::to_usize()) as isize);

		K::NR::full_unroll(&mut |j|{
			K::MR::full_unroll(&mut |i|{
				if K::FMA::to_usize() > 0 {
					ab[j][i] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[j][i]);
				} else {
					ab[j][i] = ab[j][i] + at::<K::T>(a, i) * at::<K::T>(b, j);
				}
			});
		});

	});

	K::NR::full_unroll(&mut |j|{
		K::MR::full_unroll(&mut |i|{
			ab[j][i] = ab[j][i]*alpha;
		});
	});

	// for j in 0..K::NR::to_usize() {
	// 	for i in 0..K::MR::to_usize() {
	// 		ab[j][i] = ab[j][i]*alpha;
	// 	}
	// }

	ab
}


/// prefetch locations of C which will be written too
#[inline(always)]
unsafe fn write_prefetch<K: KernelConfig>(c: *mut K::T, rsc: isize, csc: isize) {

	if rsc == 1 {
		K::NR::full_unroll(&mut |j|{
			prefetch(c.offset(csc * j as isize) as *mut i8, 1, 3, 1); // addr, write, nonlocal, data
		});	
	} else if csc == 1 {
		K::MR::full_unroll(&mut |i|{
			prefetch(c.offset(rsc * i as isize) as *mut i8, 1, 3, 1); // addr, write, nonlocal, data
		});	
	} else {
		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				prefetch(c.offset(rsc * i as isize + csc * j as isize) as *mut i8, 1, 3, 1); // addr, write, nonlocal, data
			}
		}
	}
}

/// Choose writes to C in a cache/vectorisation friendly manner if possible
#[inline(always)]
unsafe fn kernel_write<K: KernelConfig>(c: *mut K::T, rsc: isize, csc: isize, ab: & GA<GA<K::T, K::NR>, K::MR>) {

	if rsc == 1 {
		// K::MR::full_unroll(&mut |i|{
		// 	K::NR::full_unroll(&mut |j|{
		// 		let v = c.offset(1 * i as isize + csc * j as isize);
		// 		*v = *v + ab[i][j];
		// 	});	
		// });	
		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				let v = c.offset(1 * i as isize + csc * j as isize);
				*v = *v + ab[i][j];
			}
		}
	} else if csc == 1 {
		// K::MR::full_unroll(&mut |i|{
		// 	K::NR::full_unroll(&mut |j|{
		// 		let v = c.offset(rsc * i as isize + 1 * j as isize);
		// 		*v = *v + ab[i][j];
		// 	});	
		// });
		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				let v = c.offset(rsc * i as isize + 1 * j as isize);
				*v = *v + ab[i][j];
			}
		}
	} else {
		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				let v = c.offset(rsc * i as isize + csc * j as isize);
				*v = *v + ab[i][j];
			}
		}
	}

}

/// Choose writes to C in a cache/vectorisation friendly manner if possible
#[inline(always)]
unsafe fn kernel_write_trans<K: KernelConfig>(c: *mut K::T, rsc: isize, csc: isize, ab: & GA<GA<K::T, K::MR>, K::NR>) {

	if rsc == 1 {
		// K::NR::full_unroll(&mut |j|{
		// 	K::MR::full_unroll(&mut |i|{
		// 		let v = c.offset(1 * i as isize + csc * j as isize);
		// 		*v = *v + ab[j][i];
		// 	});	
		// });	
		for j in 0..K::NR::to_usize() {
			for i in 0..K::MR::to_usize() {
				let v = c.offset(1 * i as isize + csc * j as isize);
				*v = *v + ab[j][i];
			}
		}
	} else if csc == 1 {
		// K::NR::full_unroll(&mut |j|{
		// 	K::MR::full_unroll(&mut |i|{
		// 		let v = c.offset(rsc * i as isize + 1 * j as isize);
		// 		*v = *v + ab[j][i];
		// 	});	
		// });	
		for j in 0..K::NR::to_usize() {
			for i in 0..K::MR::to_usize() {
				let v = c.offset(rsc * i as isize + 1 * j as isize);
				*v = *v + ab[j][i];
			}
		}
	} else {
		for j in 0..K::NR::to_usize() {
			for i in 0..K::MR::to_usize() {
				let v = c.offset(rsc * i as isize + csc * j as isize);
				*v = *v + ab[j][i];
			}
		}
	}

}


#[inline(always)]
unsafe fn at<T: Copy>(ptr: *const T, i: usize) -> T {
	*ptr.offset(i as isize)
}

// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use typenum::Unsigned;
//use typenum_loops::Loop;
use generic_params::*;
use std::cmp::min;
//use std::mem;
use num_traits::Float;
use loops::full_unroll;
use super::prefetch;

/// Call the GEMM kernel with a "masked" output C.
///
/// Simply redirect the MR by NR kernel output to the passed
/// in `mask_buf`, and copy the non masked region to the real
/// C.
///
/// + rows: rows of kernel unmasked
/// + cols: cols of kernel unmasked
#[inline(always)]
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
#[inline(always)]
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
#[inline(never)]
unsafe fn kernel_compute<K: KernelConfig>(k: usize, alpha: K::T, a: *const K::T, b: *const K::T) -> GA<GA<K::T, K::NR>, K::MR>{

	// Compute matrix multiplication into ab[i][j]
	let mut ab = <GA<GA<K::T, K::NR>, K::MR>>::default();

	for l in 0..k {
	//K::KU::partial_unroll(k, |l, _|{
		let a = a.offset((l*K::MR::to_usize()) as isize);
		let b = b.offset((l*K::NR::to_usize()) as isize);

		// if K::NR::to_usize()*mem::size_of::<K::T>() >= 32 {prefetch(b.offset(128) as *mut i8, 0, 3, 1);}
		// if K::MR::to_usize()*mem::size_of::<K::T>() >= 32 {prefetch(a.offset(128) as *mut i8, 0, 3, 1);}

		// let mut aa = <GA<K::T, K::MR>>::default();
		// K::MR::full_unroll(|i|{
		// 	aa[i] = at::<K::T>(a, i);
		// });
		// K::MR::full_unroll(|i|{
		// 	let mut bb = <GA<K::T, K::NR>>::default();
		// 	K::NR::full_unroll(|j|{
		// 		bb[j] = at::<K::T>(b, j);
		// 	});
		// 	K::NR::full_unroll(|j|{
		// 		if K::FMA::to_usize() > 0 {
		// 			ab[i][j] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[i][j]);
		// 		} else {
		// 			//ab[i][j] += at::<K::T>(a, i) * at::<K::T>(b, j);
		// 			ab[i][j] += aa[i]*bb[j];
		// 		}
		// 	});
		// });

		// full_unroll(K::MR::to_usize(), &mut |i|{
		// 	full_unroll(K::NR::to_usize(), &mut |j|{
		// 		if K::FMA::to_usize() > 0 {
		// 			ab[i][j] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[i][j]);
		// 		} else {
		// 			ab[i][j] += at::<K::T>(a, i) * at::<K::T>(b, j);
		// 		}
		// 	});
		// });

		for i in 0..K::MR::to_usize() {
			for j in 0..K::NR::to_usize() {
				if K::FMA::to_usize() > 0 {
					ab[i][j] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[i][j]);
				} else {
					ab[i][j] += at::<K::T>(a, i) * at::<K::T>(b, j);
				}
			}
		}
	//});
	}

	// full_unroll(K::MR::to_usize(), &mut |i|{
	// 	full_unroll(K::NR::to_usize(), &mut |j|{
	// 		ab[i][j] = ab[i][j]*alpha;
	// 	});
	// });

	for i in 0..K::MR::to_usize() {
		for j in 0..K::NR::to_usize() {
			ab[i][j] = ab[i][j]*alpha;
		}
	}

	ab
}


/// Split out compute for better vectorisation
#[inline(never)]
unsafe fn kernel_compute_trans<K: KernelConfig>(k: usize, alpha: K::T, a: *const K::T, b: *const K::T) -> GA<GA<K::T, K::MR>, K::NR>{

	// Compute matrix multiplication into ab[i][j]
	let mut ab = <GA<GA<K::T, K::MR>, K::NR>>::default();

	for l in 0..k {
	//K::KU::partial_unroll(k, |l, _|{
		let a = a.offset((l*K::MR::to_usize()) as isize);
		let b = b.offset((l*K::NR::to_usize()) as isize);

		//if K::NR::to_usize()*mem::size_of::<K::T>() >= 32 {prefetch(b.offset(128) as *mut i8, 0, 3, 1);}
		//if K::MR::to_usize()*mem::size_of::<K::T>() >= 32 {prefetch(a.offset(128) as *mut i8, 0, 3, 1);}
		
		// let mut bb = <GA<K::T, K::NR>>::default();
		// K::NR::full_unroll(|j|{
		// 	bb[j] = at::<K::T>(b, j);
		// });

		// K::NR::full_unroll(|j|{

		// 	let mut aa = <GA<K::T, K::MR>>::default();
		// 	K::MR::full_unroll(|i|{
		// 		aa[i] = at::<K::T>(a, i);
		// 	});
		// 	K::MR::full_unroll(|i|{
		// 		if K::FMA::to_usize() > 0 {
		// 			ab[j][i] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[j][i]);
		// 		} else {
		// 			ab[j][i] = ab[j][i] + aa[i]*bb[j];
		// 			//ab[j][i] = ab[j][i] + at::<K::T>(a, i) * at::<K::T>(b, j);
		// 		}
		// 	});
		// });

		// full_unroll(K::NR::to_usize(), &mut |j|{
		// 	full_unroll(K::MR::to_usize(), &mut |i|{
		// 		if K::FMA::to_usize() > 0 {
		// 			ab[j][i] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[j][i]);
		// 		} else {
		// 			ab[j][i] = ab[j][i] + at::<K::T>(a, i) * at::<K::T>(b, j);
		// 		}
		// 	});
		// });

		for j in 0..K::NR::to_usize() {
			for i in 0..K::MR::to_usize() {
				if K::FMA::to_usize() > 0 {
					ab[j][i] = at::<K::T>(a, i).mul_add(at::<K::T>(b, j), ab[j][i]);
				} else {
					ab[j][i] = ab[j][i] + at::<K::T>(a, i) * at::<K::T>(b, j);
				}
			}
		}
	//});
	}

	// full_unroll(K::NR::to_usize(), &mut |j|{
	// 	full_unroll(K::MR::to_usize(), &mut |i|{
	// 		ab[j][i] = ab[j][i]*alpha;
	// 	});
	// });

	for j in 0..K::NR::to_usize() {
		for i in 0..K::MR::to_usize() {
			ab[j][i] = ab[j][i]*alpha;
		}
	}

	ab
}


/// prefetch locations of C which will be written too
#[inline(always)]
unsafe fn write_prefetch<K: KernelConfig>(c: *mut K::T, rsc: isize, csc: isize) {

	if rsc == 1 {
		full_unroll(K::NR::to_usize(), &mut |j|{
			prefetch(c.offset(csc * j as isize) as *mut i8, 1, 3, 1); // addr, write, nonlocal, data
		});	
	} else if csc == 1 {
		full_unroll(K::MR::to_usize(), &mut |i|{
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
		// full_unroll(K::MR::to_usize(), &mut |i|{
		// 	full_unroll(K::NR::to_usize(), &mut |j|{
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
		// full_unroll(K::MR::to_usize(), &mut |i|{
		// 	full_unroll(K::NR::to_usize(), &mut |j|{
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
		// full_unroll(K::NR::to_usize(), &mut |j|{
		// 	full_unroll(K::MR::to_usize(), &mut |i|{
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
		// full_unroll(K::NR::to_usize(), &mut |j|{
		// 	full_unroll(K::MR::to_usize(), &mut |i|{
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

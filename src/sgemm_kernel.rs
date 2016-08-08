// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use kernel::GemmKernel;
use archparam;

pub enum Gemm { }

pub type T = f32;

const MR: usize = 4;
const NR: usize = 16;

macro_rules! loopMR {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
//        let $i = 4; $e;
//        let $i = 5; $e;
//        let $i = 6; $e;
//        let $i = 7; $e;

        assert!($i == MR -1);
    }}
}

macro_rules! loopNR {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
        let $i = 4; $e;
        let $i = 5; $e;
        let $i = 6; $e;
        let $i = 7; $e;
        
        let $i = 8; $e;
        let $i = 9; $e;
        let $i = 10; $e;
        let $i = 11; $e;
        let $i = 12; $e;
        let $i = 13; $e;
        let $i = 14; $e;
        let $i = 15; $e;

//        let $i = 16; $e;
//        let $i = 17; $e;
//        let $i = 18; $e;
//        let $i = 19; $e;
//        let $i = 20; $e;
//        let $i = 21; $e;
//        let $i = 22; $e;
//        let $i = 23; $e;
        assert!($i == NR -1);
    }}
}

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn align_to() -> usize {
        0
    }

    #[inline(always)]
    fn mr() -> usize {
        MR
    }
    #[inline(always)]
    fn nr() -> usize {
        NR
    }

    #[inline(always)]
    fn always_masked() -> bool {
        false
    }

    #[inline(always)]
    fn nc() -> usize {
        ((archparam::S_NC + NR - 1) / NR) * NR
    }
    #[inline(always)]
    fn kc() -> usize {
        archparam::S_KC
    }
    #[inline(always)]
    fn mc() -> usize {
        ((archparam::S_MC + MR - 1) / MR) * MR
    }

    #[inline(always)]
    unsafe fn kernel(k: usize,
                     alpha: T,
                     a: *const T,
                     b: *const T,
                     c: *mut T,
                     rsc: isize,
                     csc: isize) {
        kernel(k, alpha, a, b, c, rsc, csc)
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
pub unsafe fn kernel(k: usize,
                     alpha: T,
                     a: *const T,
                     b: *const T,
                     c: *mut T,
                     rsc: isize,
                     csc: isize) {
    let mut ab = [[0.; NR]; MR];

    kernel_compute(k, alpha, a, b, &mut ab);

    kernel_write(c, rsc, csc, &ab);

}

/// Split out compute for better vectorisation
#[inline(never)]
unsafe fn kernel_compute(k: usize, alpha: T, a: *const T, b: *const T, ab_: &mut [[T; NR]; MR]) {
    if k > 0 {
	    let mut ab = *ab_;
	    loopMR!(i, loopNR!(j, ab[i][j] = 0.0)); // this removes the loads from stack, and xorps the registers
	    let mut a = a;
	    let mut b = b;
	
	    // Compute matrix multiplication into ab[i][j]
	    // Due to llvm/MIR update a temporary array is no longer needed for vectorisation, and unroll doesnt ruin register allocation
		unroll_by_4!(k, {
			loopMR!(i, loopNR!(j, ab[i][j] += at(a, i) * at(b, j)));
			
	        a = a.offset(MR as isize);
	        b = b.offset(NR as isize);
		});
	    
	    loopMR!(i, loopNR!(j, ab[i][j] *= alpha));
	
	    *ab_ = ab;    	
    }
}

/// Choose writes to C in a cache/vectorisation friendly manner
#[inline(always)]
unsafe fn kernel_write(c: *mut T, rsc: isize, csc: isize, ab: &[[T; NR]; MR]) {

    if rsc == 1 {
        for j in 0..NR {
            loopMR!(i, *c.offset(1 * i as isize + csc * j as isize) += ab[i][j]);
        }
    } else if csc == 1 {
        for i in 0..MR {
            loopNR!(j, *c.offset(rsc * i as isize + 1 * j as isize) += ab[i][j]);
        }
    } else {
        for i in 0..MR {
            for j in 0..NR {
                *c.offset(rsc * i as isize + csc * j as isize) += ab[i][j];
            }
        }
    }

}


#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

// #[test] TODO: Needs to be updated for general MR/NR
// fn test_gemm_kernel() {
//    let mut a = [1.; 16];
//    let mut b = [0.; 32];
//    for (i, x) in a.iter_mut().enumerate() {
//        *x = i as f32;
//    }
//
//    for i in 0..4 {
//        b[i + i * 8] = 1.;
//    }
//    let mut c = [0.; 16];
//    unsafe {
//        kernel(4, 1., &a[0], &b[0], 0., &mut c[0], 1, 4);
//        // col major C
//    }
//    assert_eq!(&a, &c);
// }

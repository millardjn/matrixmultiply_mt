use generic_params::*;
use typenum::*;
use gemm;
use typenum::{U4, U2};


type FS = unsafe fn(usize, usize, usize, f32, *const f32, isize, isize, *const f32, isize, isize, f32, *mut f32, isize, isize, bool);
type FD = unsafe fn(usize, usize, usize, f64, *const f64, isize, isize, *const f64, isize, isize, f64, *mut f64, isize, isize, bool);

static THIN_SGEMMS: [&'static FS; 16] = [
		&(gemm::gemm_loop::<SgemmCache, S32x1t> as FS), // 0
		&(gemm::gemm_loop::<SgemmCache, S32x1t> as FS), // 1
		&(gemm::gemm_loop::<SgemmCache, S24x2t> as FS), // 2
		&(gemm::gemm_loop::<SgemmCache, S16x3t> as FS), // 3
		&(gemm::gemm_loop::<SgemmCache, S16x4t> as FS), // 4
		&(gemm::gemm_loop::<SgemmCache, S16x5t> as FS), // 5
		&(gemm::gemm_loop::<SgemmCache, S16x3t> as FS), // 6
		&(gemm::gemm_loop::<SgemmCache, S8x7t> as FS), // 7
		&(gemm::gemm_loop::<SgemmCache, S16x4t> as FS), // 8
		&(gemm::gemm_loop::<SgemmCache, S16x3t> as FS), // 9
		&(gemm::gemm_loop::<SgemmCache, S16x5t> as FS), // 10
		&(gemm::gemm_loop::<SgemmCache, S16x4t> as FS), // 11 (+1)
		&(gemm::gemm_loop::<SgemmCache, S16x4t> as FS), // 12
		&(gemm::gemm_loop::<SgemmCache, S16x5t> as FS), // 13 (+2)
		&(gemm::gemm_loop::<SgemmCache, S16x5t> as FS), // 14 (+1)
		&(gemm::gemm_loop::<SgemmCache, S16x5t> as FS), // 15
	];

static THIN_DGEMMS: [&'static FD; 16] = [
		&(gemm::gemm_loop::<DgemmCache, D16x1t> as FD), // 0
		&(gemm::gemm_loop::<DgemmCache, D16x1t> as FD), // 1
		&(gemm::gemm_loop::<DgemmCache, D8x2t> as FD), // 2
		&(gemm::gemm_loop::<DgemmCache, D8x3t> as FD), // 3
		&(gemm::gemm_loop::<DgemmCache, D8x4t> as FD), // 4
		&(gemm::gemm_loop::<DgemmCache, D8x5t> as FD), // 5
		&(gemm::gemm_loop::<DgemmCache, D8x3t> as FD), // 6
		&(gemm::gemm_loop::<DgemmCache, D4x7t> as FD), // 7
		&(gemm::gemm_loop::<DgemmCache, D8x4t> as FD), // 8
		&(gemm::gemm_loop::<DgemmCache, D8x3t> as FD), // 9
		&(gemm::gemm_loop::<DgemmCache, D8x5t> as FD), // 10
		&(gemm::gemm_loop::<DgemmCache, D8x4t> as FD), // 11 (+1)
		&(gemm::gemm_loop::<DgemmCache, D8x4t> as FD), // 12
		&(gemm::gemm_loop::<DgemmCache, D8x5t> as FD), // 13 (+2)
		&(gemm::gemm_loop::<DgemmCache, D8x5t> as FD), // 14 (+1)
		&(gemm::gemm_loop::<DgemmCache, D8x5t> as FD), // 15
	];

pub unsafe fn sgemm(m: usize,
					k: usize,
					n: usize,
					alpha: f32,
					a: *const f32,
					rsa: isize,
					csa: isize,
					b: *const f32,
					rsb: isize,
					csb: isize,
					beta: f32,
					c: *mut f32,
					rsc: isize,
					csc: isize,
					multithread: bool) {

	if n < THIN_SGEMMS.len() {
		THIN_SGEMMS[n](m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
		return;
	}

	if n > 28 && csc == 1 {
		gemm::gemm_loop::<SgemmCache, S4x16>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
	} else {
		gemm::gemm_loop::<SgemmCache, S16x4t>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
	}

}

/// General matrix multiplication (f64)
///
/// C ← α A B + β C
///
/// + m, k, n: dimensions
/// + a, b, c: pointer to the first element in the matrix
/// + A: m by k matrix
/// + B: k by n matrix
/// + C: m by n matrix
/// + rs<em>x</em>: row stride of *x*
/// + cs<em>x</em>: col stride of *x*
///
/// Strides for A and B may be arbitrary. Strides for C must not result in
/// elements that alias each other, for example they can not be zero.
///
/// If β is zero, then C does not need to be initialized.
pub unsafe fn dgemm(m: usize,
					k: usize,
					n: usize,
					alpha: f64,
					a: *const f64,
					rsa: isize,
					csa: isize,
					b: *const f64,
					rsb: isize,
					csb: isize,
					beta: f64,
					c: *mut f64,
					rsc: isize,
					csc: isize,
					multithread: bool) {
	
	if n < THIN_SGEMMS.len() {
		THIN_DGEMMS[n](m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
		return;
	}

	if n > 28 && csc == 1 {
		gemm::gemm_loop::<DgemmCache, D4x8>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
	} else {
		gemm::gemm_loop::<DgemmCache, D8x4t>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, multithread);
	}
}


#[allow(unused)]
pub struct S8x8; // 8 avx registers
impl KernelConfig for S8x8 {
	type T = f32;
	type MR = U8;
	type NR = U8;
	type KU = U5;
	type TR = U0;
	type FMA = U0;
}

#[allow(unused)]
pub struct S8x8t; // 8 avx registers
impl KernelConfig for S8x8t {
	type T = f32;
	type MR = U8;
	type NR = U8;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

pub struct S4x16; // 8 avx registers
impl KernelConfig for S4x16 {
	type T = f32;
	type MR = U4;
	type NR = U16;
	type KU = U5;
	type TR = U0;
	type FMA = U0;
}

// Thin Kernels
pub struct S8x7t; // 7 avx registers
impl KernelConfig for S8x7t {
	type T = f32;
	type MR = U8;
	type NR = U7;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

#[allow(unused)]
pub struct S16x6t; // 12 avx registers
impl KernelConfig for S16x6t {
	type T = f32;
	type MR = U16;
	type NR = U6;
	type KU = U4;
	type TR = U1;
	type FMA = U0;
}

pub struct S16x5t; // 10 avx registers
impl KernelConfig for S16x5t {
	type T = f32;
	type MR = U16;
	type NR = U5;
	type KU = U4;
	type TR = U1;
	type FMA = U0;
}

pub struct S16x4t; // 8 avx registers
impl KernelConfig for S16x4t {
	type T = f32;
	type MR = U16;
	type NR = U4;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

pub struct S16x3t; // 6 avx registers
impl KernelConfig for S16x3t {
	type T = f32;
	type MR = U16;
	type NR = U3;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}

pub struct S24x2t; // 6 avx registers
impl KernelConfig for S24x2t {
	type T = f32;
	type MR = U24;
	type NR = U2;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}

pub struct S32x1t;// 4 avx registers
impl KernelConfig for S32x1t {
	type T = f32;
	type MR = U32;
	type NR = U1;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}




#[allow(unused)]
pub struct D8x4; // 8 avx registers
impl KernelConfig for D8x4 {
	type T = f64;
	type MR = U8;
	type NR = U4;
	type KU = U5;
	type TR = U0;
	type FMA = U0;
}

#[allow(unused)]
pub struct D4x8t; // 8 avx registers
impl KernelConfig for D4x8t {
	type T = f64;
	type MR = U4;
	type NR = U8;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

pub struct D4x8; // 8 avx registers
impl KernelConfig for D4x8 {
	type T = f64;
	type MR = U4;
	type NR = U8;
	type KU = U5;
	type TR = U0;
	type FMA = U0;
}




// Thin Kernels
pub struct D4x7t; // 7 avx registers
impl KernelConfig for D4x7t {
	type T = f64;
	type MR = U4;
	type NR = U7;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

#[allow(unused)]
pub struct D8x6t; // 12 avx registers
impl KernelConfig for D8x6t {
	type T = f64;
	type MR = U8;
	type NR = U6;
	type KU = U4;
	type TR = U1;
	type FMA = U0;
}

pub struct D8x5t; // 10 avx registers
impl KernelConfig for D8x5t {
	type T = f64;
	type MR = U8;
	type NR = U5;
	type KU = U4;
	type TR = U1;
	type FMA = U0;
}

pub struct D8x4t; // 8 avx registers
impl KernelConfig for D8x4t {
	type T = f64;
	type MR = U8;
	type NR = U4;
	type KU = U5;
	type TR = U1;
	type FMA = U0;
}

pub struct D8x3t; // 6 avx registers
impl KernelConfig for D8x3t {
	type T = f64;
	type MR = U8;
	type NR = U3;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}

pub struct D8x2t; // 4 avx registers
impl KernelConfig for D8x2t {
	type T = f64;
	type MR = U8;
	type NR = U2;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}

pub struct D16x1t;// 4 avx registers
impl KernelConfig for D16x1t {
	type T = f64;
	type MR = U16;
	type NR = U1;
	type KU = U8;
	type TR = U1;
	type FMA = U0;
}



pub struct SgemmCache;
impl CacheConfigValues for SgemmCache{
	type A = U64;
	type MT = U128;
	type MC = U64;
	type NC = U1024;
	type KC = U256;
}

pub struct DgemmCache;
impl CacheConfigValues for DgemmCache{
	type A = U64;
	type MT = U128;
	type MC = U32;
	type NC = U512;
	type KC = U256;
}

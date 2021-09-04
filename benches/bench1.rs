#![feature(test)]
extern crate matrixmultiply_mt;

extern crate test;

// Compute GFlop/s
// by flop / s =  2 x 2 M N K / time

/// + A: m by k matrix
/// + B: k by n matrix
/// + C: m by n matrix
macro_rules! mat_mul {
    ($modname:ident, $gemm:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use test::{Bencher};
            use matrixmultiply_mt::$gemm;
            $(
            #[bench]
            fn $name(bench: &mut Bencher)
            {
                let a = vec![0.; $m * $k];
                let b = vec![0.; $k * $n];
                let mut c = vec![0.; $m * $n];
                bench.iter(|| {
                    unsafe {
                    	// all row major
                        $gemm(
                            $m, $k, $n,
                            1.,
                            a.as_ptr(), $k, 1,
                            b.as_ptr(), $n, 1,
                            0.,
                            c.as_mut_ptr(), $n, 1,
                            );

						// all col major
                        $gemm(
                            $m, $k, $n,
                            1.,
                            a.as_ptr(), 1, $m,
                            b.as_ptr(), 1, $k,
                            0.,
                            c.as_mut_ptr(), 1, $m,
                            );
                    }
                });
            }
            )+
        }
    };
}

// name, m, n, k
mat_mul! {mat_mul_f32, sgemm,
	(m0004, 4, 4, 4)
	(m0005, 5, 5, 5)
	(m0006, 6, 6, 6)
	(m0007, 7, 7, 7)
	(m0008, 8, 8, 8)
	(m0009, 9, 9, 9)
	(m0012, 12, 12, 12)
	(m0016, 16, 16, 16)
	(m0032, 32, 32, 32)
	(m0064, 64, 64, 64)
	(m0127, 127, 127, 127)
	(m0256, 256, 256, 256)
	(m0512, 512, 512, 512)
	(mix16x4, 32, 4, 32)
	(mix32x2, 32, 2, 32)
	(mix97, 97, 97, 125)

	(skew1024x01, 1024,  1, 64)
	(skew1024x02, 1024,  2, 64)
	(skew1024x03, 1024,  3, 64)
	(skew1024x04, 1024,  4, 64)
	(skew1024x05, 1024,  5, 64)
	(skew1024x06, 1024,  6, 64)
	(skew1024x07, 1024,  7, 64)
	(skew1024x08, 1024,  8, 64)
	(skew1024x09, 1024,  9, 64)
	(skew1024x10, 1024, 10, 64)
	(skew1024x11, 1024, 11, 64)
	(skew1024x12, 1024, 12, 64)
	(skew1024x13, 1024, 13, 64)
	(skew1024x14, 1024, 14, 64)
	(skew1024x15, 1024, 15, 64)
	(skew1024x16, 1024, 16, 64)
	(skew1024x17, 1024, 17, 64)
	(skew16x1024, 16, 1024, 64)

	(mix128x10000x128, 128, 10000, 128)
}

mat_mul! {mat_mul_f32_st, sgemm_st,
	(m0004, 4, 4, 4)
	(m0005, 5, 5, 5)
	(m0006, 6, 6, 6)
	(m0007, 7, 7, 7)
	(m0008, 8, 8, 8)
	(m0009, 9, 9, 9)
	(m0012, 12, 12, 12)
	(m0016, 16, 16, 16)
	(m0032, 32, 32, 32)
	(m0064, 64, 64, 64)
	(m0127, 127, 127, 127)
	(m0256, 256, 256, 256)
	(m0512, 512, 512, 512)
	(mix16x4, 32, 4, 32)
	(mix32x2, 32, 2, 32)
	(mix97, 97, 97, 125)

	(skew1024x01, 1024,  1, 64)
	(skew1024x02, 1024,  2, 64)
	(skew1024x03, 1024,  3, 64)
	(skew1024x04, 1024,  4, 64)
	(skew1024x05, 1024,  5, 64)
	(skew1024x06, 1024,  6, 64)
	(skew1024x07, 1024,  7, 64)
	(skew1024x08, 1024,  8, 64)
	(skew1024x09, 1024,  9, 64)
	(skew1024x10, 1024, 10, 64)
	(skew1024x11, 1024, 11, 64)
	(skew1024x12, 1024, 12, 64)
	(skew1024x13, 1024, 13, 64)
	(skew1024x14, 1024, 14, 64)
	(skew1024x15, 1024, 15, 64)
	(skew1024x16, 1024, 16, 64)
	(skew1024x17, 1024, 17, 64)
	(skew16x1024, 16, 1024, 64)

	(mix128x10000x128, 128, 10000, 128)
}

mat_mul! {mat_mul_f64, dgemm,
	(m004, 4, 4, 4)
	(m007, 7, 7, 7)
	(m008, 8, 8, 8)
	(m012, 12, 12, 12)
	(m016, 16, 16, 16)
	(m032, 32, 32, 32)
	(m064, 64, 64, 64)
	(m127, 127, 127, 127)
	(m256, 256, 256, 256)
	(m512, 512, 512, 512)
	(mix16x4, 32, 4, 32)
	(mix32x2, 32, 2, 32)
	(mix97, 97, 97, 125)
	(skew256x32, 256, 16, 512)
	(skew32x256, 16, 256, 512)
	(mix128x10000x128, 128, 10000, 128)
}

mat_mul! {mat_mul_f64_st, dgemm_st,
	(m004, 4, 4, 4)
	(m007, 7, 7, 7)
	(m008, 8, 8, 8)
	(m012, 12, 12, 12)
	(m016, 16, 16, 16)
	(m032, 32, 32, 32)
	(m064, 64, 64, 64)
	(m127, 127, 127, 127)
	(m256, 256, 256, 256)
	(m512, 512, 512, 512)
	(mix16x4, 32, 4, 32)
	(mix32x2, 32, 2, 32)
	(mix97, 97, 97, 125)
	(skew256x32, 256, 16, 512)
	(skew32x256, 16, 256, 512)
	(mix128x10000x128, 128, 10000, 128)
}

use std::ops::{Add, Mul};

trait Z {
	fn zero() -> Self;
}
impl Z for f32 {
	fn zero() -> Self {
		0.
	}
}
impl Z for f64 {
	fn zero() -> Self {
		0.
	}
}

// simple, slow, correct (hopefully) mat mul (Row Major)
#[inline(never)]
fn reference_mat_mul<A>(m: usize, k: usize, n: usize, a: &[A], b: &[A], c: &mut [A])
where
	A: Z + Add<Output = A> + Mul<Output = A> + Copy,
{
	assert!(a.len() >= m * k);
	assert!(b.len() >= k * n);
	assert!(c.len() >= m * n);

	for i in 0..m {
		for j in 0..n {
			unsafe {
				let celt = c.get_unchecked_mut(i * m + j);
				*celt = (0..k).fold(A::zero(), move |s, x| {
					s + *a.get_unchecked(i * k + x) * *b.get_unchecked(x * n + j)
				});
			}
		}
	}
}

macro_rules! ref_mat_mul {
    ($modname:ident, $ty:ty, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use test::{Bencher};
            use super::reference_mat_mul;
            $(
            #[bench]
            fn $name(bench: &mut Bencher)
            {
                let a = vec![0. as $ty; $m * $n];
                let b = vec![0.; $n * $k];
                let mut c = vec![0.; $m * $k];
                bench.iter(|| {
                    reference_mat_mul($m, $n, $k, &a, &b, &mut c);
                    c[0]
                });
            }
            )+
        }
    };
}
ref_mat_mul! {ref_mat_mul_f32, f32,
	(m004, 4, 4, 4)
	(m005, 5, 5, 5)
	(m006, 6, 6, 6)
	(m007, 7, 7, 7)
	(m008, 8, 8, 8)
	(m009, 9, 9, 9)
	(m012, 12, 12, 12)
	(m016, 16, 16, 16)
	(m032, 32, 32, 32)
	(m064, 64, 64, 64)
}

// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use num::Num;
use typenum::*;
use typenum_loops::Loop;
use generic_array::{GenericArray, ArrayLength};

pub type GA<T, U> = GenericArray<T, U>;

pub struct SgemmKernelAVX;
impl KernelConfig for SgemmKernelAVX {
	type T = f32;
	type MR = U4;
	type NR = U16;
	type KU = U5;
}

pub struct DgemmKernelAVX;
impl KernelConfig for DgemmKernelAVX {
	type T = f64;
	type MR = U4;
	type NR = U8;
	type KU = U5;
}

pub struct SgemmCache;
impl CacheConfig for SgemmCache{
	type A = U64;
	type MC = U64;
	type NC = U1024;
	type KC = U256;
}

pub struct DgemmCache;
impl CacheConfig for DgemmCache{
	type A = U64;
	type MC = U64;
	type NC = U512;
	type KC = U256;
}

pub trait KernelConfig: 'static {
	/// Matrix element: [f32|f64]
	type T: Element;
	/// Number of registers used in kernel
	type MR: Unsigned + Loop + ArrayLength<GA<Self::T, Self::NR>>;
	/// Register width
	type NR: Unsigned + Loop + ArrayLength<Self::T>;
	/// Unrolling factor of kernel loop over K dimension.
	type KU: Unsigned + Loop + ArrayLength<Self::T>;
}

pub trait CacheConfig: 'static {
	/// Required alignment in bytes. Usually cache line size
	type A: Unsigned;

	/// Rows of Ap at a time. (3rd loop)
	///
	/// Cuts Ap into A0, A1, .., Ai, .. A_MC
	///
	/// Ai is packed into A~.
	///
	/// Size of A~ is KC x MC
	type MC: Unsigned;

	/// Columns in C, B that we handle at a time. (5th loop)
	///
	/// Cuts B into B0, B1, .. Bj, .. B_NC
	type NC: Unsigned;

	/// Rows of Bj at a time (4th loop)
	///
	/// Columns of A at a time.
	///
	/// Cuts A into Ap
	///
	/// Cuts Bj into Bp, which is packed into B~.
	///
	/// Size of B~ is NC x KC
	type KC: Unsigned;
}


pub trait Element: Copy + Send + Default + Num {}
impl<T: Copy + Send + Default + Num> Element for T {}
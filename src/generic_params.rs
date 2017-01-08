// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use num_traits::float::Float;
use typenum::*;
use typenum_loops::Loop;
use generic_array::{GenericArray, ArrayLength};

pub type GA<T, U> = GenericArray<T, U>;

pub struct SgemmAVX16x4;
impl KernelConfig for SgemmAVX16x4 {
	type T = f32;
	type MR = U4;
	type NR = U16;
	type KU = U5;
	type R = SgemmAVX8x8;
}

pub struct SgemmAVX8x8;
impl KernelConfig for SgemmAVX8x8 {
	type T = f32;
	type MR = U8;
	type NR = U8;
	type KU = U5;
	type R = SgemmAVX4x8;
}

pub struct SgemmAVX4x8;
impl KernelConfig for SgemmAVX4x8 {
	type T = f32;
	type MR = U8;
	type NR = U4;
	type KU = U6;
	type R = Self;
}

pub struct DgemmAVX8x4;
impl KernelConfig for DgemmAVX8x4 {
	type T = f64;
	type MR = U4;
	type NR = U8;
	type KU = U5;
	type R = DgemmAVX4x8;
}

pub struct DgemmAVX4x8;
impl KernelConfig for DgemmAVX4x8 {
	type T = f64;
	type MR = U8;
	type NR = U4;
	type KU = U5;
	type R = DgemmAVX2x8;
}

pub struct DgemmAVX2x8;
impl KernelConfig for DgemmAVX2x8 {
	type T = f64;
	type MR = U8;
	type NR = U2;
	type KU = U4;
	type R = Self;
}

pub struct SgemmCache;
impl CacheConfig for SgemmCache{
	type A = U64;
	type MT = U64;
	type MC = U64;
	type NC = U1024;
	type KC = U256;
}

pub struct DgemmCache;
impl CacheConfig for DgemmCache{
	type A = U64;
	type MT = U64;
	type MC = U32;
	type NC = U512;
	type KC = U256;
}

// pub struct DgemmCache;
// impl CacheConfig for DgemmCache{
// 	type A = U64;
// 	type MT = U64;
// 	type MC = U64;
// 	type NC = U1024;
// 	type KC = U128;
// }

pub trait KernelConfig: 'static {

	/// Matrix element: [f32|f64]
	type T: Element;
	/// Number of registers used in kernel
	type MR: Unsigned + Loop + ArrayLength<GA<Self::T, Self::NR>>;
	/// Register width
	type NR: Unsigned + Loop + ArrayLength<Self::T>;
	/// Unrolling factor of kernel loop over K dimension.
	type KU: Unsigned + Loop;
	/// Alternative kernel with lower NR. Set to Self at end of chain.
	type R: KernelConfig; // <T=Self::T> constraint solver cant handle this yet
}

pub trait CacheConfig: 'static {
	/// Required alignment in bytes. Usually cache line size
	type A: Unsigned;

	/// how much smaller than MC*NC*KC does a multiply have to be before multithreading should be disallowed
	type MT: Unsigned;

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


pub trait Element: Copy + Send + Default + Float {}
impl<T: Copy + Send + Default + Float> Element for T {}
// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// IMPLEMENTED
/// sandybridge    - Select the sandybridge processor.
/// genericsse2    - For older CPUs
///
/// DEFERRED
/// ivybridge      - Select the ivybridge processor. DEFER TO SANDYBRIDGE
///
/// UN-IMPLEMENTED
/// amdfam10       - Select the amdfam10 processor.
/// athlon         - Select the athlon processor.
/// athlon-4       - Select the athlon-4 processor.
/// athlon-fx      - Select the athlon-fx processor.
/// athlon-mp      - Select the athlon-mp processor.
/// athlon-tbird   - Select the athlon-tbird processor.
/// athlon-xp      - Select the athlon-xp processor.
/// athlon64       - Select the athlon64 processor.
/// athlon64-sse3  - Select the athlon64-sse3 processor.
/// atom           - Select the atom processor.
/// barcelona      - Select the barcelona processor.
/// bdver1         - Select the bdver1 processor.
/// bdver2         - Select the bdver2 processor.
/// bdver3         - Select the bdver3 processor.
/// bdver4         - Select the bdver4 processor.
/// bonnell        - Select the bonnell processor.
/// broadwell      - Select the broadwell processor.
/// btver1         - Select the btver1 processor.
/// btver2         - Select the btver2 processor.
/// c3             - Select the c3 processor.
/// c3-2           - Select the c3-2 processor.
/// cannonlake     - Select the cannonlake processor.
/// core-avx-i     - Select the core-avx-i processor.
/// core-avx2      - Select the core-avx2 processor.
/// core2          - Select the core2 processor.
/// corei7         - Select the corei7 processor.
/// corei7-avx     - Select the corei7-avx processor.
/// generic        - Select the generic processor.
/// geode          - Select the geode processor.
/// haswell        - Select the haswell processor.
/// i386           - Select the i386 processor.
/// i486           - Select the i486 processor.
/// i586           - Select the i586 processor.
/// i686           - Select the i686 processor.
/// k6             - Select the k6 processor.
/// k6-2           - Select the k6-2 processor.
/// k6-3           - Select the k6-3 processor.
/// k8             - Select the k8 processor.
/// k8-sse3        - Select the k8-sse3 processor.
/// knl            - Select the knl processor.
/// lakemont       - Select the lakemont processor.
/// nehalem        - Select the nehalem processor.
/// nocona         - Select the nocona processor.
/// opteron        - Select the opteron processor.
/// opteron-sse3   - Select the opteron-sse3 processor.
/// penryn         - Select the penryn processor.
/// pentium        - Select the pentium processor.
/// pentium-m      - Select the pentium-m processor.
/// pentium-mmx    - Select the pentium-mmx processor.
/// pentium2       - Select the pentium2 processor.
/// pentium3       - Select the pentium3 processor.
/// pentium3m      - Select the pentium3m processor.
/// pentium4       - Select the pentium4 processor.
/// pentium4m      - Select the pentium4m processor.
/// pentiumpro     - Select the pentiumpro processor.
/// prescott       - Select the prescott processor.
/// silvermont     - Select the silvermont processor.
/// skx            - Select the skx processor.
/// skylake        - Select the skylake processor.
/// skylake-avx512 - Select the skylake-avx512 processor.
/// slm            - Select the slm processor.
/// westmere       - Select the westmere processor.
/// winchip-c6     - Select the winchip-c6 processor.
/// winchip2       - Select the winchip2 processor.
/// x86-64         - Select the x86-64 processor.
/// yonah          - Select the yonah processor.
/// znver1         - Select the znver1 processor.


use num_traits::float::Float;
use std::cmp;
use typenum::*;
use typenum_loops::Loop;
use generic_array::{GenericArray, ArrayLength};

pub type GA<T, U> = GenericArray<T, U>;

pub trait KernelConfig: 'static {

	/// Matrix element: [f32|f64]
	type T: Element;
	/// Number of registers used in kernel
	type MR: Unsigned + Loop + ArrayLength<Self::T> + ArrayLength<GA<Self::T, Self::NR>>;
	/// Register width
	type NR: Unsigned + Loop + ArrayLength<Self::T> + ArrayLength<GA<Self::T, Self::MR>>;
	/// Unrolling factor of kernel loop over K dimension.
	type KU: Unsigned + Loop;
	/// Trans Flag, if TR > 0 then the kernel will be implemented such that MR sized registers are used.
	type TR: Unsigned;
	/// Fused multiply add Flag, if FMA > 0 then the kernel will be implemented with fused ops.
	type FMA: Unsigned;
}

pub trait CacheConfigValues: 'static {
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

pub trait CacheConfig<K: KernelConfig>: CacheConfigValues{
	fn alignment() -> usize{
		Self::A::to_usize()
	}
	fn multithread_factor() -> usize{
		Self::MT::to_usize()
	}
	fn mc() -> usize{
		cmp::max(1, Self::MC::to_usize()/K::MR::to_usize()) * K::MR::to_usize()
	}
	fn nc() -> usize{
		cmp::max(1, Self::NC::to_usize()/K::NR::to_usize()) * K::NR::to_usize()
	}
	fn kc() -> usize{
		Self::KC::to_usize()
	}
}

impl<T: CacheConfigValues, K: KernelConfig> CacheConfig<K> for T{}

pub trait Element: Copy + Send + Default + Float {}
impl<T: Copy + Send + Default + Float> Element for T {}

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

pub struct S4x4;
impl KernelConfig for S4x4 {
	type T = f32;
	type MR = U4;
	type NR = U4;
	type KU = U4;
	type TR = U0;
	type FMA = U0;
}

pub struct D2x4;
impl KernelConfig for D2x4 {
	type T = f64;
	type MR = U4;
	type NR = U2;
	type KU = U4;
	type TR = U0;
	type FMA = U0;
}

pub struct S4x4fma;
impl KernelConfig for S4x4fma {
	type T = f32;
	type MR = U4;
	type NR = U4;
	type KU = U4;
	type TR = U0;
	type FMA = U1;
}

pub struct D2x4fma;
impl KernelConfig for D2x4fma {
	type T = f64;
	type MR = U4;
	type NR = U2;
	type KU = U4;
	type TR = U0;
	type FMA = U1;
}
// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use threadpool::ThreadPool;
use num_cpus;
use std::sync::{Condvar, Mutex};
use std::sync::atomic::{Ordering, AtomicUsize};
use std::cmp::{min, max};
use std::mem::size_of;
use std::mem::align_of;
use util::range_chunk;
use util::round_up_to;
use util::round_up_div;

use rawpointer::PointerExt;
use typenum::Unsigned;
use generic_params::*;
use generic_kernel;
use generic_array::ArrayLength;
use typenum_loops::Loop;
use num_traits::identities::{One, Zero};
use {sse_stmxcsr, sse_ldmxcsr};
use snb_kernels;
use hwl_kernels;
use super::prefetch;

//use std::intrinsics::atomic_singlethreadfence;

/// If 'ftz_daz' feature is not enabled this does nothing and returns 0.
/// Sets the ftz and daz bits of the mxcsr register, zeroing input and result subnormal floats.
/// Returns the current mxcsr register value, which should be restored at a later time by reset_ftz_and_daz();
fn set_ftz_and_daz() -> i32 {
	//if cfg!(ftz_daz) {
		let mut old_mxcsr = 0i32;
		unsafe{sse_stmxcsr(&mut old_mxcsr as *mut i32 as *mut i8)};
		let mut new_mxcsr = old_mxcsr | 0x8040;
		unsafe{sse_ldmxcsr(&mut new_mxcsr as *mut i32 as *mut i8)};
		//unsafe{atomic_singlethreadfence()};// prevent this being moved backward after floating point operations
		old_mxcsr
	// } else {
	// 	0
	// }
}

/// If 'ftz_daz' feature is not enabled this does nothing.
/// Set the mxcsr register back to its previous value
fn reset_ftz_and_daz(mut old_mxcsr: i32){
	//if cfg!(ftz_daz) {
		//unsafe{atomic_singlethreadfence()};// prevent this being moved forward infront of floating point operations
		unsafe{sse_ldmxcsr(&mut old_mxcsr as *mut i32 as *mut i8)};
	//}
}


lazy_static! {
	static ref NUM_CPUS: usize = num_cpus::get();
	static ref THREAD_POOL: Mutex<ThreadPool> = Mutex::new(ThreadPool::new(*NUM_CPUS));
}

/// General matrix multiplication (f32)
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
					csc: isize) {
	if k == 0 || m == 0 || n == 0 {
		return;
	}
	let (m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc) = if n > m {
			(n, k, m, b, csb, rsb, a, csa, rsa, c, csc, rsc)
		} else {
			(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
		};

	if cfg!(arch_haswell) {
		hwl_kernels::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
	} else if cfg!(arch_sandybridge) {
		snb_kernels::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
	} else if cfg!(arch_penryn) {
		unimplemented!();
	} else if cfg!(arch_generic4x4fma) {
		gemm_loop::<SgemmCache, S4x4fma>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
	} else {//arch_generic4x4
		gemm_loop::<SgemmCache, S4x4>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
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
					csc: isize) {
	if k == 0 || m == 0 || n == 0 {return;
	}
	let (m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc) = if n > m {
			(n, k, m, b, csb, rsb, a, csa, rsa, c, csc, rsc)
		} else {
			(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
		};

	if cfg!(arch_haswell) {
		hwl_kernels::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
	} else if cfg!(arch_sandybridge) {
		snb_kernels::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
	} else if cfg!(arch_penryn) {
		unimplemented!();
	} else if cfg!(arch_generic4x4fma) {
		gemm_loop::<DgemmCache, D2x4fma>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
	} else {//arch_generic4x4
		gemm_loop::<DgemmCache, D2x4>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
	}
}

/// split M direction over multiple CPUS as long as the work units wont be too small.
///
/// First, maximise num_threads.
/// Second, maximise cmc.
/// while preserving 3 things:
/// * dont let `cmc*min(C:KC, k)*min(C:NC, n)` go below `C:MC*C:KC*C:NC/C:MT` avoid excessive synchronisation costs
/// * `cmc` must be divisible by `kmr`
/// * dont let `cmc*min(C:KC, k)` go above `C:MC*C:KC` still fit in cache
/// returns (num_threads, cmc)
#[cfg(no_multithreading)]
fn get_num_threads_and_cmc<C: CacheConfig<K>, K: KernelConfig>(_m: usize, _k: usize, _n: usize) -> (usize, usize){
	(1, C::mc())
}

/// split M direction over multiple CPUS as long as the work units wont be too small.
///
/// First, maximise num_threads.
/// Second, maximise cmc.
/// while preserving 3 things:
/// * dont let `cmc*min(C:KC, k)*min(C:NC, n)` go below `C:MC*C:KC*C:NC/C:MT` avoid excessive synchronisation costs
/// * `cmc` must be divisible by `kmr`
/// * dont let `cmc*min(C:KC, k)` go above `C:MC*C:KC` still fit in cache
/// returns (num_threads, cmc)
#[cfg(not(no_multithreading))]
fn get_num_threads_and_cmc<C: CacheConfig<K>, K: KernelConfig>(m: usize, k: usize, n: usize) -> (usize, usize){

	let m_bands = round_up_div(m, K::MR::to_usize());

	// maximum bound mc is that the ~A block must be less than the maximum size
	let max_mc_bands = {
		let max_size = C::kc() * C::mc();
		let max_kc = max(min(k, C::kc()), 1);
		(max_size / max_kc + C::mc())/(K::MR::to_usize()*2)
	};

	// minimum bound on mc before splitting over threads is some fraction of the max compute per ~A cache block
	let min_split_mc_bands = min(max_mc_bands, {
		let max_compute = C::mc() * C::kc() * C::nc();
		let min_compute = max(max_compute/C::multithread_factor(), 1);

		let max_kc = max(min(k, C::kc()), 1);
		let max_nc = max(min(round_up_to(n, K::NR::to_usize()), C::nc()), 1);

		round_up_div(round_up_div(min_compute, max_nc * max_kc), K::MR::to_usize())
	});


	let num_threads = {
		let full_blocks = m_bands/min_split_mc_bands;
		max(min(*NUM_CPUS, full_blocks), 1)
	};


	let mc = {
		let m_bands_per_thread = max(round_up_div(m_bands, num_threads), 1);
		let blocks_per_thread = round_up_div(m_bands_per_thread, max_mc_bands);
		let mc_bands = min(round_up_div(m_bands_per_thread, blocks_per_thread), m_bands);
		let mc = mc_bands * K::MR::to_usize();

		// normally both of these would be true. however when dealing with low compute high memory operations, the mc direction may be split over multiple cpus when the cache limit is exceeded, even when mc will be less than minimum splitting size for compute
		//debug_assert!(num_threads == 1 || mc >= min_split_mc_bands* K::MR::to_usize(), "threads{} mc{} min{} max{} bpt{} mbpt{}", num_threads, mc, min_split_mc_bands, max_mc_bands, blocks_per_thread, m_bands_per_thread);
		//debug_assert!(num_threads == 1 || mc*max(min(k, C::KC::to_usize()), 1)*max(min(round_up_to(n, K::NR::to_usize()), C::NC::to_usize()), 1) >= C::MC::to_usize() * C::KC::to_usize() * C::NC::to_usize()/C::MT::to_usize());
		mc
	};

	debug_assert!(mc <= max_mc_bands* K::MR::to_usize(), "mc{} min{} max{}", mc, min_split_mc_bands, max_mc_bands);
	debug_assert!(num_threads <= *NUM_CPUS);
	debug_assert_eq!(0, mc % K::MR::to_usize());

	(num_threads, mc)
}

/// Implement matrix multiply using packed buffers and a microkernel strategy
/// The type parameter `K` is the gemm microkernel configuration.
/// The type parameter `C` is the outer gemm cache blocking configuration.
pub unsafe fn gemm_loop<C: CacheConfig<K>, K: KernelConfig>(m: usize,
					   k: usize,
					   n: usize,
					   alpha: K::T,
					   a: *const K::T,
					   rsa: isize,
					   csa: isize,
					   b: *const K::T,
					   rsb: isize,
					   csb: isize,
					   beta: K::T,
					   c: *mut K::T,
					   rsc: isize,
					   csc: isize)
{
	
	debug_assert!(m * n == 0 || (rsc != 0 && csc != 0));
	let knr = K::NR::to_usize();
	let kmr = K::MR::to_usize();
	let cnc = C::nc();
	let ckc = C::kc();
	let (num_threads, cmc) = get_num_threads_and_cmc::<C, K>(m, k, n);

	assert_eq!(0, cnc % knr);
	assert_eq!(0, cmc % kmr);
	
	let pool_opt = if num_threads > 1 {
		THREAD_POOL.lock().ok()
	} else {
		None
	};

	// must be able to achieve alignment using only elementwise offsets
	// size_of returns size + alignment padding.
	assert!(C::alignment() % size_of::<K::T>() == 0); 
	let (_vec , app_stride, app_base, bpp) = aligned_packing_vec::<K, C::A>(m, k, n, cmc, ckc, cnc, num_threads);
	debug_assert_eq!(bpp as usize % align_of::<K::T>(), 0);

	// LOOP 5: split n into nc parts
	for (l5, nc) in range_chunk(n, cnc) {
		dprint!("LOOP 5, {}, nc={}", l5, nc);
		let b = b.stride_offset(csb, cnc * l5);
		let c = c.stride_offset(csc, cnc * l5);

		// LOOP 4: split k in kc parts
		for (l4, kc) in range_chunk(k, ckc) {
			dprint!("LOOP 4, {}, kc={}", l4, kc);
			let b = b.stride_offset(rsb, ckc * l4);
			let a = a.stride_offset(csa, ckc * l4);
			debug!(for elt in &mut packv {
				*elt = K::T::one();
			});

			// Pack B -> B~
			pack::<K::T, K::NR>(kc, nc, knr, bpp, b, csb, rsb);

			if let (Some(pool), true) = (pool_opt.as_ref(), num_threads > 1) {
				// Need a struct to smuggle pointers across threads. ugh!
				struct Ptrs<T: Element> {
					app: *mut T,
					bpp: *mut T,
					a: *const T,
					c: *mut T,
					loop_counter: *mut AtomicUsize,
					sync: *mut (Mutex<bool>, Condvar, AtomicUsize),
				}
				unsafe impl<T: Element> Send for Ptrs<T> {}

				// Threads decrement the atomic int and move on to other work, last thread out flips the mutex/condvar
				// This is likely a useless micro optimisation, but might be useful if the threadpool is large & shared & stressed and workloads are small?
				let mut sync = (Mutex::new(false), Condvar::new(), AtomicUsize::new(num_threads));
				let mut loop_counter = AtomicUsize::new(0);

				for cpu_id in 0..num_threads {
					let p = Ptrs::<K::T> {
						app: app_base.offset(app_stride * cpu_id as isize),
						bpp: bpp,
						a: a,
						c: c,
						loop_counter: &mut loop_counter as *mut _,
						sync: &mut sync as *mut _,
					};
					debug_assert_eq!(p.app as usize % align_of::<K::T>(), 0);

					pool.execute(move || {
						let bpp = p.bpp;
						let app = p.app;
						let a = p.a;
						let c = p.c;
						let (ref lock, ref cvar, ref thread_counter) = *p.sync;

						let mut next_id = (*p.loop_counter).fetch_add(1, Ordering::Relaxed);
						let mxcsr = set_ftz_and_daz();
						// LOOP 3: split m into mc parts
						for (l3, mc) in range_chunk(m, cmc) {
							
							if l3 < next_id {continue;}

							dprint!("LOOP 3, {}, mc={}, id={}", l3, mc);
							let a = a.stride_offset(rsa, cmc * l3);
							let c = c.stride_offset(rsc, cmc * l3);

							// Pack A -> A~
							pack::<K::T, K::MR>(kc, mc, kmr, app, a, rsa, csa);

							// First time writing to C, use user's `beta`, else accumulate
							let betap = if l4 == 0 {beta} else {K::T::one()};

							// LOOP 2 and 1
							gemm_packed::<K>(nc, kc, mc, alpha, app, bpp, betap, c, rsc, csc);

							next_id = (*p.loop_counter).fetch_add(1, Ordering::Relaxed);
						}
						
						let x = thread_counter.fetch_sub(1, Ordering::AcqRel);
						if x == 1 {
							*lock.lock().unwrap() = true;
							cvar.notify_all();
						}
						reset_ftz_and_daz(mxcsr);
					});
				  
				}

				let (ref lock, ref cvar, ref thread_counter) = sync;
				let mut finished = lock.lock().unwrap();
				while !*finished {
					finished = cvar.wait(finished).unwrap();
				}
				debug_assert!(thread_counter.load(Ordering::SeqCst) == 0);
			} else {
				let app = app_base;
				let mxcsr = set_ftz_and_daz();
				for (l3, mc) in range_chunk(m, cmc) {

					dprint!("LOOP 3, {}, mc={}", l3, mc);
					let a = a.stride_offset(rsa, cmc * l3);
					let c = c.stride_offset(rsc, cmc * l3);

					// Pack A -> A~
					pack::<K::T, K::MR>(kc, mc, kmr, app, a, rsa, csa);

					// First time writing to C, use user's `beta`, else accumulate
					let betap = if l4 == 0 {beta} else {<K::T>::one()};

					// LOOP 2 and 1
					gemm_packed::<K>(nc, kc, mc, alpha, app, bpp, betap, c, rsc, csc);
				}
				reset_ftz_and_daz(mxcsr);
			}

		}
	}
}

/// Loops 1 and 2 around the µ-kernel
///
/// + app: packed A (A~)
/// + bpp: packed B (B~)
/// + nc: columns of packed B
/// + kc: columns of packed A / rows of packed B
/// + mc: rows of packed A
unsafe fn gemm_packed<K: KernelConfig>(nc: usize,
						 kc: usize,
						 mc: usize,
						 alpha: K::T,
						 app: *const K::T,
						 bpp: *const K::T,
						 beta: K::T,
						 c: *mut K::T,
						 rsc: isize,
						 csc: isize)
{
	let mr = K::MR::to_usize();
	let nr = K::NR::to_usize();

	// Zero or prescale if necessary
	if beta.is_zero() {
		zero_block::<K::T>(mc, nc, c, rsc, csc);
	} else if beta != K::T::one() {
		scale_block::<K::T>(beta, mc, nc, c, rsc, csc);
	}

	// LOOP 2: through micropanels in packed `b`
	for (l2, nr_) in range_chunk(nc, nr) {
		let bpp = bpp.stride_offset(1, kc * nr * l2);
		let c = c.stride_offset(csc, nr * l2);

		// LOOP 1: through micropanels in packed `a` while `b` is constant
		for (l1, mr_) in range_chunk(mc, mr) {
			let app = app.stride_offset(1, kc * mr * l1);
			let c = c.stride_offset(rsc, mr * l1);

			// GEMM KERNEL
			if nr_ < nr || mr_ < mr {
				generic_kernel::masked_kernel::<K>(kc, alpha, &*app, &*bpp, &mut *c, rsc, csc, mr_, nr_);
			} else {
				generic_kernel::kernel::<K>(kc, alpha, app, bpp, c, rsc, csc);
			}
		}
	}
}

unsafe fn scale_block<T: Element>(beta: T,
									 rows: usize,
									 cols: usize,
									 c: *mut T,
									 rsc: isize,
									 csc: isize) {

	if rsc == 1 {
		for col in 0..cols {
			for row in 0..rows {
				let cptr = c.offset(1 * row as isize + csc * col as isize);
				*cptr = *cptr * beta;
			}
		}
	} else if csc == 1 {
		for row in 0..rows {
			for col in 0..cols {
				let cptr = c.offset(rsc * row as isize + 1 * col as isize);
				*cptr = *cptr * beta;
			}
		}
	} else {
		for col in 0..cols {
			for row in 0..rows {
				let cptr = c.offset(rsc * row as isize + csc * col as isize);
				*cptr = *cptr * beta;
			}
		}
	}
}

unsafe fn zero_block<T: Element>(rows: usize,
									cols: usize,
									c: *mut T,
									rsc: isize,
									csc: isize) {

	if rsc == 1 {
		for col in 0..cols {
			for row in 0..rows {
				let cptr = c.offset(1 * row as isize + csc * col as isize);
				*cptr = T::zero();
			}
		}
	} else if csc == 1 {
		for row in 0..rows {
			for col in 0..cols {
				let cptr = c.offset(rsc * row as isize + 1 * col as isize);
				*cptr = T::zero();
			}
		}
	} else {
		for col in 0..cols {
			for row in 0..rows {
				let cptr = c.offset(rsc * row as isize + csc * col as isize);
				*cptr = T::zero();
			}
		}
	}
}

/// Allocate a vector of uninitialized data to be used for both B~ and multiple A~ packing buffers.
///
/// + A~ needs be KC x MC x num_a
/// + B~ needs be KC x NC
/// but we can make them smaller if the matrix is smaller than this (just ensure
/// we have rounded up to a multiple of the kernel size).
///
/// Returns an uninitialised packing vector, stride between of each app region, aligned pointer to start of first app, and aligned pointer to start of b
unsafe fn aligned_packing_vec<K: KernelConfig, A: Unsigned>(m: usize, k: usize, n: usize, cmc: usize, ckc: usize, cnc: usize, num_a: usize) -> (Vec<K::T>, isize, *mut K::T, *mut K::T){
	let m = min(m, cmc);
	let k = min(k, ckc);
	let n = min(n, cnc);

	let align = A::to_usize();
	// round up k, n to multiples of mr, nr
	// round up to multiple of kc
	
	assert!(align % size_of::<K::T>() == 0); // size_of is size + alignment padding
	//assert!(align % align_of::<K::T>() == 0);
	
	let align_elems = align / size_of::<K::T>();

	let apack_size = k * round_up_to(m, K::MR::to_usize());
	let bpack_size = k * round_up_to(n, K::NR::to_usize());

	let padding_bytes1 = align_elems; // give room to let first A~ be aligned
	let padding_bytes2 = if align_elems == 0 {0} else {round_up_to(apack_size, align_elems) - apack_size}; // room after each A~ to keep next section aligned
	let nelem = padding_bytes1 + (apack_size + padding_bytes2) * num_a + bpack_size;
	
	let mut v = Vec::with_capacity(nelem);
	v.set_len(nelem);
	
	
	dprint!("packed nelem={}, apack={}, bpack={},
			 m={} k={} n={}",
			nelem,
			apack_size,
			bpack_size,
			m,
			k,
			n);

	
	let mut a_ptr = v.as_mut_ptr();
	if align != 0 {
		let current_misalignment = a_ptr as usize % align;
		debug_assert!(current_misalignment % size_of::<K::T>() == 0); // check that a whole number of elements are required to re-align the pointer
		if current_misalignment != 0 {
			a_ptr = a_ptr.offset(((align - current_misalignment) / size_of::<K::T>()) as isize);
		}
	}
	
	let b_ptr = a_ptr.offset(((apack_size + padding_bytes2)*num_a) as isize);
	
	(v, (apack_size + padding_bytes2) as isize, a_ptr, b_ptr)
}


/// Pack matrix into `pack`
/// Variable notation refers to packing ~A. for ~B mr = NR::to_usize
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
/// + zero: zero element to pad with
unsafe fn pack<T: Element, MR: Loop + ArrayLength<T>>(kc: usize,
				  mc: usize,
				  mr: usize,
				  pack: *mut T,
				  a: *const T,
				  rsa: isize,
				  csa: isize)
{
	
	debug_assert_eq!(mr, MR::to_usize());
	
	if csa == 1 {
		part_pack_row_major::<T, MR>(kc, mc, mr, pack, a, rsa, csa);
	} else if rsa == 1 {
		part_pack_col_major::<T, MR>(kc, mc, mr, pack, a, rsa, csa);
	} else {
		part_pack_strided::<T, MR>(kc, mc, mr, pack, a, rsa, csa);
	}

	let rest = mc % mr;
	if rest > 0 {
		part_pack_end::<T, MR>(kc, mc, mr, pack, a, rsa, csa, rest);
	}
}

/// Pack matrix into `pack`
/// Only packs whole micro panels, and must only be called when csa == 1 (row_major format)
/// Variable notation refers to packing ~A. for ~B mr = NR::to_usize
/// 
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
/// + zero: zero element to pad with
//#[inline(never)]
unsafe fn part_pack_row_major<T: Element, MR: Loop + ArrayLength<T>>(kc: usize,
				  mc: usize,
				  mr: usize,
				  pack: *mut T,
				  a: *const T,
				  rsa: isize,
				  csa: isize)
{
	
	debug_assert_eq!(mr, MR::to_usize());
	debug_assert_eq!(csa, 1);
	let csa = 1isize;
	let mr = MR::to_usize();

	for ir in 0..mc / mr {

		let a = a.offset((ir * mr) as isize * rsa);
		let pack = pack.offset((ir * mr * kc) as isize);

		// prefetch the rows of a for the panel ahead of the current one,
		// but only once we are a finite number of elements away from the next panel
		let kc_prefetch = kc.saturating_sub(128/mr); // 64 and 128 seem to work well on sandybridge

		for j in 0..kc_prefetch{
			let a = a.stride_offset(csa, j);
			MR::full_unroll(|i|{
				*(pack.offset((j*mr+i)as isize)) = *a.stride_offset(rsa, i);
			});
		}

		MR::full_unroll(|i|{
			prefetch(a.offset(((ir+1) * mr + i) as isize * rsa) as *mut i8, 0, 3, 1);
		});

		for j in kc_prefetch..kc{
			let a = a.stride_offset(csa, j);
			MR::full_unroll(|i|{
				*(pack.offset((j*mr+i)as isize)) = *a.stride_offset(rsa, i);
			});
		}
	}
}

/// Pack matrix into `pack`
/// Only packs whole micro panels, and must only be called when rsa == 1 (col_major format)
/// Variable notation refers to packing ~A. for ~B mr = NR::to_usize
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
//#[inline(never)]
unsafe fn part_pack_col_major<T: Element, MR: Loop + ArrayLength<T>>(kc: usize,
				  mc: usize,
				  mr: usize,
				  pack: *mut T,
				  a: *const T,
				  rsa: isize,
				  csa: isize)
{
	debug_assert_eq!(mr, MR::to_usize());
	debug_assert_eq!(rsa, 1);
	let rsa = 1isize;
	let mr = MR::to_usize();

	for ir in 0..mc / mr {
		let a = a.offset((ir * mr) as isize * rsa);
		let pack = pack.offset((ir * mr * kc) as isize);
		prefetch(a.offset(((ir+1) * mr) as isize * rsa) as *mut i8, 0, 3, 1);

		for j in 0..kc{
			prefetch(a.stride_offset(csa, j+64/mr) as *mut i8, 0, 3, 1);

			let mut arr = <GA<T, MR>>::default();
			let a = a.stride_offset(csa, j);
			MR::full_unroll(|i|{
				arr[i] = *a.stride_offset(rsa, i);
			});

			MR::full_unroll(|i|{
				*(pack.offset((j*mr+i)as isize)) = arr[i];
			});
		}
	}
}

/// Pack matrix into `pack`
/// Only packs whole micro panels, can handle any rsa or csa
/// Variable notation refers to packing ~A. for ~B mr = NR::to_usize
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
#[cold]
unsafe fn part_pack_strided<T: Element, MR: Loop + ArrayLength<T>>(kc: usize,
				  mc: usize,
				  mr: usize,
				  pack: *mut T,
				  a: *const T,
				  rsa: isize,
				  csa: isize)
{
	
	debug_assert_eq!(mr, MR::to_usize());
	let mr = MR::to_usize();

	for ir in 0..mc / mr {
		let a = a.offset((ir * mr) as isize * rsa);
		let pack = pack.offset((ir * mr * kc) as isize);
		for j in 0..kc{
			MR::full_unroll(|i|{
				*(pack.offset((j*mr+i)as isize)) = *a.stride_offset(rsa, i).stride_offset(csa, j);
			});
		}
	}
}

/// Pack matrix into `pack`
/// Only packs the last partial micro panel, can handle any rsa or csa
/// Variable notation refers to packing ~A. for ~B mr = NR::to_usize
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
#[cold]
unsafe fn part_pack_end<T: Element, MR: Loop + ArrayLength<T>>(kc: usize,
				  mc: usize,
				  mr: usize,
				  pack: *mut T,
				  a: *const T,
				  rsa: isize,
				  csa: isize,
				  rest: usize)
{
	debug_assert_eq!(mr, MR::to_usize());
	let mr = MR::to_usize();

	let mut pack = pack.offset(((mc/mr)*mr*kc) as isize);

	// Pad with zeros to multiple of kernel size (uneven mc)
	let row_offset = mc - rest;//(mc / mr) * mr;
	for j in 0..kc {
		MR::full_unroll(|i|{
			if i < rest {
				*pack = *a.stride_offset(rsa, i + row_offset).stride_offset(csa, j);
			} else {
				*pack = T::zero();
			}
			pack.inc();
		});
	}
}

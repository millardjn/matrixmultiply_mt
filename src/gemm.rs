// Original work Copyright 2016 bluss
// Modified work Copyright 2016 J. Millard.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate num_cpus;
extern crate threadpool;

use threadpool::ThreadPool;
use std::sync::{Condvar, Mutex};
use std::sync::atomic::{Ordering, AtomicUsize};
use std::cmp::{min, max};
use std::mem::size_of;
use std::mem::align_of;

use util::range_chunk;
use util::round_up_to;

use pointer::PointerExt;
use typenum::{Unsigned, U4};

use generic_params::*;
use generic_kernel;
use typenum_loops::Loop;

use num::{Zero, One};

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

	gemm::<SgemmCache, SgemmKernelAVX>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)

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

	gemm::<DgemmCache, DgemmKernelAVX>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
}

/// General matrix multiplication (f64|f32)
/// The type parameter `K` is the gemm microkernel configuration.
/// The type parameter `C` is the outer gemm cache blocking configuration.

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
pub unsafe fn gemm<C: CacheConfig, K: KernelConfig>(m: usize,
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

	if m >= K::NR::to_usize() || n >= K::NR::to_usize() || k >= K::NR::to_usize() {
		let (same_threads, _) = get_num_threads_and_mc(m, K::MR::to_usize(), C::MC::to_usize());
		let (flip_threads, _) = get_num_threads_and_mc(n, K::MR::to_usize(), C::MC::to_usize());

		// Flip matrix multiply to make maximise multithreadingon rectangular C, otherwise flip if it makes iteration over C more cache friendly
		// Tradeoff between multithreading and iteration order could be made more intelligently, such as, if max cores already available, then choose to optimise access patterns.
		let (m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc) = if flip_threads > same_threads || (flip_threads == same_threads && rsc.abs() < csc.abs()) {
				(n, k, m, b, csb, rsb, a, csa, rsa, c, csc, rsc)
			} else {
				(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
			};
		gemm_loop::<C, K>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
	} else {
		// Each case for for row stride or col stride == 1 should be special cased.
		// Currently only very small matricies are handled here, eventually thin matrices (low reuse) should be too.
		if beta.is_zero() {
			for i in 0..m as isize {
				for j in 0..n as isize {
					let celt = c.offset((i * rsc + j*csc) );
					*celt = (0..k as isize).fold(K::T::zero(),
						move |s, x| s + *a.offset(i * rsa + x * csa) * *b.offset(x * rsb + j * csb) * alpha);
				}
			}
		} else {
			for i in 0..m as isize {
				for j in 0..n as isize {
					let celt = c.offset((i * rsc + j*csc) );
					*celt = (0..k as isize).fold(*celt*beta,
						move |s, x| s + *a.offset(i * rsa + x * csa) * *b.offset(x * rsb + j * csb) * alpha);
				}
			}
		}
	}
}

/// rough adaption to split M direction over multiple CPUS if the work units wont be too small
/// this could be improved to ensure each thread gets an equal number of chunks
/// Rules: Don't go over cmc_, dont go under cmc_*2/3, stay divisible by kmr
fn get_num_threads_and_mc(m: usize, mr: usize, mc: usize) -> (usize, usize){
	let cmc_ = ((max( /// Take the max of either:
					min((m + *NUM_CPUS - 1) / *NUM_CPUS, mc), // 1. The min of round_up(m/numcpus) or the default cmc
					max((mc*2)/3, mr) // 2. The max of half the default cmc, or the kernel size in the m direction
				)+ mr - 1) / mr) * mr; // Then round up by kernel size

	let num_m_chunks = (m + cmc_ - 1) / cmc_;
	let num_threads = min(num_m_chunks, *NUM_CPUS);
	(num_threads, cmc_)
}

/// Implement matrix multiply using packed buffers and a microkernel strategy
/// The type parameter `K` is the gemm microkernel configuration.
/// The type parameter `C` is the outer gemm cache blocking configuration.
unsafe fn gemm_loop<C: CacheConfig, K: KernelConfig>(m: usize,
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
	let cnc = C::NC::to_usize();
	let ckc = C::KC::to_usize();
	let (num_threads, cmc) = get_num_threads_and_mc(m, kmr, C::MC::to_usize());
	debug_assert_eq!(0, cmc % kmr);
	debug_assert_eq!(0, cnc % knr);

	
	let pool_opt = if num_threads > 1 {
		THREAD_POOL.lock().ok()
	} else {
		None
	};

	let (_vec , app_stride, app_base, bpp) = aligned_packing_vec::<C, K>(m, k, n, num_threads);
	
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
				*elt = <_>::one();
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
					pair: *mut (Mutex<bool>, Condvar, AtomicUsize),
				}
				unsafe impl<T: Element> Send for Ptrs<T> {}

				// Threads decrement the atomic int and move on to other work, last thread out flips the mutex/condvar
				// This is likely a useless micro optimisation, but might be useful if the threadpool is large & shared & stressed and workloads are small?
				let mut pair = (Mutex::new(false), Condvar::new(), AtomicUsize::new(num_threads));    

				for cpu_id in 0..num_threads {
					let p = Ptrs::<K::T> {
						app: app_base.offset(app_stride * cpu_id as isize),
						bpp: bpp,
						a: a,
						c: c,
						pair: &mut pair as *mut _,
					};

					pool.execute(move || {
						let bpp = p.bpp;
						let app = p.app;
						let a = p.a;
						let c = p.c;
						let &mut(ref lock, ref cvar, ref counter) = p.pair.as_mut().unwrap();


						// LOOP 3: split m into mc parts
						for (l3, mc) in range_chunk(m, cmc) {
							if l3 % num_threads != cpu_id {continue;} // threads leapfrog each other

							dprint!("LOOP 3, {}, mc={}", l3, mc);
							let a = a.stride_offset(rsa, cmc * l3);
							let c = c.stride_offset(rsc, cmc * l3);

							// Pack A -> A~
							pack::<K::T, K::MR>(kc, mc, kmr, app, a, rsa, csa);

							// First time writing to C, use user's `beta`, else accumulate
							let betap = if l4 == 0 {beta} else {<K::T>::one()};

							// LOOP 2 and 1
							gemm_packed::<C, K>(nc, kc, mc, alpha, app, bpp, betap, c, rsc, csc);
						}

						let x = counter.fetch_sub(1, Ordering::Relaxed);
						if x == 1 {
							*lock.lock().unwrap() = true;
							cvar.notify_one();
						}

					});
				  
				}

				let &(ref lock, ref cvar, _) = &pair;
				let mut finished = lock.lock().unwrap();
				while !*finished {
					finished = cvar.wait(finished).unwrap();
				}

			} else {
				let app = app_base;
				for (l3, mc) in range_chunk(m, cmc) {

					dprint!("LOOP 3, {}, mc={}", l3, mc);
					let a = a.stride_offset(rsa, cmc * l3);
					let c = c.stride_offset(rsc, cmc * l3);

					// Pack A -> A~
					pack::<K::T, K::MR>(kc, mc, kmr, app, a, rsa, csa);

					// First time writing to C, use user's `beta`, else accumulate
					let betap = if l4 == 0 {beta} else {<K::T>::one()};

					// LOOP 2 and 1
					gemm_packed::<C, K>(nc, kc, mc, alpha, app, bpp, betap, c, rsc, csc);
				}
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
unsafe fn gemm_packed<C: CacheConfig, K: KernelConfig>(nc: usize,
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
/// Return packing vector, stride between of each app region, aligned pointer to start of first app, and aligned pointer to start of b
unsafe fn aligned_packing_vec<C: CacheConfig, K: KernelConfig>(m: usize, k: usize, n: usize, num_a: usize) -> (Vec<K::T>, isize, *mut K::T, *mut K::T){
	let m = min(m, C::MC::to_usize());
	let k = min(k, C::KC::to_usize());
	let n = min(n, C::NC::to_usize());
	let align_to = C::A::to_usize();
	// round up k, n to multiples of mr, nr
	// round up to multiple of kc
	
	debug_assert!(align_to % size_of::<K::T>() == 0);
	debug_assert!(align_to % align_of::<K::T>() == 0);
	
	let align_elems = align_to / size_of::<K::T>();

	let apack_size = k * round_up_to(m, K::MR::to_usize());
	let bpack_size = k * round_up_to(n, K::NR::to_usize());

	let align1 = align_elems; // give room to let first A~ be aligned
	let align2 = if align_elems == 0 {0} else {round_up_to(apack_size, align_elems) - apack_size}; // room after each A~ to keep next section aligned
	let nelem = align1 + (apack_size + align2) * num_a + bpack_size;
	
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
	if align_to != 0 {
		let cur_align = a_ptr as usize % align_to;
		debug_assert!(cur_align % size_of::<K::T>() == 0);
		if cur_align != 0 {
			a_ptr = a_ptr.offset(((align_to - cur_align) / size_of::<K::T>()) as isize);
		}
	}
	
	let b_ptr = a_ptr.offset(((apack_size + align2)*num_a) as isize);
	
	(v, (apack_size + align2) as isize, a_ptr, b_ptr)
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
unsafe fn pack<T: Element, MR: Loop>(kc: usize,
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
		let a = a.stride_offset(rsa, ir * mr);
		let pack = pack.offset((ir * mr * kc) as isize);
		if csa == 1 {
			U4::partial_unroll(kc,|j|{
				MR::full_unroll(|i|{
					let pack = pack.offset((j*mr+i)as isize);
					*pack = *a.stride_offset(rsa, i)
							.stride_offset(1, j);
				});
			});
		} else if rsa == 1 {
			U4::partial_unroll(kc,|j|{
				MR::full_unroll(|i|{
					let pack = pack.offset((j*mr+i)as isize);
					*pack = *a.stride_offset(1, i)
							.stride_offset(csa, j);
				});
			});
		} else {
			U4::partial_unroll(kc,|j|{
				MR::full_unroll(|i|{
					let pack = pack.offset((j*mr+i)as isize);
					*pack = *a.stride_offset(rsa, i)
							.stride_offset(csa, j);
				});
			});
		}
	}

	let mut pack = pack.offset(((mc/mr)*mr*kc) as isize);

	let zero = <_>::zero();

	// Pad with zeros to multiple of kernel size (uneven mc)
	let rest = mc % mr;
	if rest > 0 {
		let row_offset = (mc / mr) * mr;
		//for j in 0..kc {
		U4::partial_unroll(kc,|j|{
			MR::full_unroll(|i|{
			//for i in 0..mr {
				if i < rest {
					*pack = *a.stride_offset(rsa, i + row_offset)
							  .stride_offset(csa, j);
				} else {
					*pack = zero;
				}
				pack.inc();
			});
		});
	}
}

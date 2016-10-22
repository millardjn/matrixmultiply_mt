// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate num_cpus;
extern crate threadpool;

use threadpool::ThreadPool;
use std::sync::{Mutex, Arc, Barrier};

use std::cmp::{min, max};
use std::mem::size_of;
use std::mem::align_of;

use util::range_chunk;
use util::round_up_to;

use kernel::GemmKernel;
use kernel::Element;
//use sgemm_kernel;
use dgemm_kernel;
use pointer::PointerExt;

use unroll::*;
use tuneable_sgemm;

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


	let (m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc) = if n > m {
			(n, k, m, b, csb, rsb, a, csa, rsa, c, csc, rsc)
		} else {
			(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
		};
	
	
    type NR = Unroll8<f32>;
    type MR = Unroll8<NR>;

	if n >= NR::val() || k >= NR::val() {

        gemm_loop::<tuneable_sgemm::Gemm<MR, NR>>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)

		//gemm_loop::<sgemm_kernel::Gemm>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
	} else {
		
		for mi in 0..m as isize{
			let c = c.offset(mi*rsc);			
			for ni in 0..n as isize{
				let c = c.offset(ni*csc);
				if beta.is_zero() {
					*c = 0.0;
				} else {
					*c *= beta
				}
			}
		}

        for mi in 0..m as isize{
            let a = a.offset(mi*rsa);
            let c = c.offset(mi*rsc);	
            for ni in 0..n as isize{
                let b = b.offset(ni*csb);
                let c = c.offset(ni*csc);
                for ki in 0..k as isize {
                    let a = a.offset(ki*csa);
                    let b = b.offset(ki*rsb);

					*c += (*a)*(*b)*alpha;
				}
			}
		}


	}
    

	// if m*n < NR*NR then use naive 3loop multiply
	
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

	let (m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc) = if n > m {
			(n, k, m, b, csb, rsb, a, csa, rsa, c, csc, rsc)
		} else {
			(m, k, n, a, rsa, csa, b, rsb, csb, c, rsc, csc)
		};    
		
    gemm_loop::<dgemm_kernel::Gemm>(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
}

const MASK_SIZE: usize = 96;
/// Ensure that GemmKernel parameters are supported
/// (alignment, microkernel size).
///
/// This function is optimized out for a supported configuration.
#[inline(always)]
fn ensure_kernel_params<K>()
    where K: GemmKernel
{
    let mr = K::mr();
    let nr = K::nr();
    assert!(mr > 0);
    assert!(nr > 0);
    assert!(mr * nr <= MASK_SIZE);
//    assert!(K::align_to() <= 32);
//    // one row/col of the kernel is limiting the max align we can provide
//    let max_align = size_of::<K::Elem>() * min(mr, nr);
//    assert!(K::align_to() <= max_align);
}

/// Implement matrix multiply using packed buffers and a microkernel
/// strategy, the type parameter `K` is the gemm microkernel.
unsafe fn gemm_loop<K>(m: usize,
                       k: usize,
                       n: usize,
                       alpha: K::Elem,
                       a: *const K::Elem,
                       rsa: isize,
                       csa: isize,
                       b: *const K::Elem,
                       rsb: isize,
                       csb: isize,
                       beta: K::Elem,
                       c: *mut K::Elem,
                       rsc: isize,
                       csc: isize)
    where K: GemmKernel
{

    debug_assert!(m * n == 0 || (rsc != 0 && csc != 0));
    let knc = K::nc();
    let kkc = K::kc();
    // let kmc = K::mc();
    // rough adaption, this can be improved to ensure each thread gets an equal number of chunks
    // todo, might need to be fixed, does it preserve mc being a multiple of mr? ((archparam::S_MC + MR - 1) / MR) * MR
    let kmc = ((max(
			    	min((m + *NUM_CPUS - 1) / *NUM_CPUS, K::mc()),
	                max(K::mc() / 2, K::mr())
				)+ K::mr() - 1) / K::mr()) * K::mr();

    let num_m_chunks = (m + kmc - 1) / kmc;
    let num_threads = min(num_m_chunks, *NUM_CPUS);
    let pool_opt = if num_threads > 1 {
        THREAD_POOL.lock().ok()
    } else {
        None
    };

    ensure_kernel_params::<K>();
	

    let (_vec , app_stride, app_base, bpp) = aligned_packing_vec::<K>(m, k, n, num_threads, K::align_to());
    
    // LOOP 5: split n into nc parts
    for (l5, nc) in range_chunk(n, knc) {
        dprint!("LOOP 5, {}, nc={}", l5, nc);
        let b = b.stride_offset(csb, knc * l5);
        let c = c.stride_offset(csc, knc * l5);

        // LOOP 4: split k in kc parts
        for (l4, kc) in range_chunk(k, kkc) {
            dprint!("LOOP 4, {}, kc={}", l4, kc);
            let b = b.stride_offset(rsb, kkc * l4);
            let a = a.stride_offset(csa, kkc * l4);
            debug!(for elt in &mut packv {
                *elt = <_>::one();
            });

            // Pack B -> B~
            pack(kc, nc, K::nr(), bpp, b, csb, rsb);

            // Need a struct to smuggle pointers across threads. ugh!
            struct Ptrs<K: GemmKernel> {
                app: *mut K::Elem,
                bpp: *mut K::Elem,
                a: *const K::Elem,
                c: *mut K::Elem,
            }
            unsafe impl<K: GemmKernel> Send for Ptrs<K> {}

            let barrier = Arc::new(Barrier::new(num_threads));
            for cpu_id in 0..num_threads {

                let p = Ptrs::<K> {
                    app: app_base.offset(app_stride * cpu_id as isize),
                    bpp: bpp,
                    a: a,
                    c: c,
                };
                let barrier = barrier.clone();

                let work = move || {
                    let bpp = p.bpp;
                    let app = p.app;
                    let a = p.a;
                    let c = p.c;


                    // LOOP 3: split m into mc parts
                    for (l3, mc) in range_chunk(m, kmc) {
                        if l3 % num_threads != cpu_id {
                            continue;
                        } // threads leapfrog each other

                        dprint!("LOOP 3, {}, mc={}", l3, mc);
                        let a = a.stride_offset(rsa, kmc * l3);
                        let c = c.stride_offset(rsc, kmc * l3);

                        // Pack A -> A~
                        pack(kc, mc, K::mr(), app, a, rsa, csa);

                        // First time writing to C, use user's `beta`, else accumulate
                        let betap = if l4 == 0 {
                            beta
                        } else {
                            <_>::one()
                        };

                        // LOOP 2 and 1
                        gemm_packed::<K>(nc, kc, mc, alpha, app, bpp, betap, c, rsc, csc);
                    }
                    barrier.wait();
                };

                if let (Some(pool), true) = (pool_opt.as_ref(), cpu_id < num_threads - 1) {
                    pool.execute(work);
                } else {
                    work();
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
unsafe fn gemm_packed<K>(nc: usize,
                         kc: usize,
                         mc: usize,
                         alpha: K::Elem,
                         app: *const K::Elem,
                         bpp: *const K::Elem,
                         beta: K::Elem,
                         c: *mut K::Elem,
                         rsc: isize,
                         csc: isize)
    where K: GemmKernel
{
    let mr = K::mr();
    let nr = K::nr();

    // Zero or prescale if necessary
    if beta.is_zero() {
        zero_block::<K>(mc, nc, c, rsc, csc);
    } else if !beta.is_one() {
        scale_block::<K>(beta, mc, nc, c, rsc, csc);
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
            if K::always_masked() || nr_ < nr || mr_ < mr {
                masked_kernel::<_, K>(kc, alpha, &*app, &*bpp, &mut *c, rsc, csc, mr_, nr_);
            } else {
                K::kernel(kc, alpha, app, bpp, c, rsc, csc);
            }
        }
    }
}

#[inline(never)]
unsafe fn scale_block<K: GemmKernel>(beta: K::Elem,
                                     rows: usize,
                                     cols: usize,
                                     c: *mut K::Elem,
                                     rsc: isize,
                                     csc: isize) {

    if rsc == 1 {
        for col in 0..cols {
            for row in 0..rows {
                let cptr = c.offset(1 * row as isize + csc * col as isize);
                (*cptr).scale_by(beta);
            }
        }
    } else if csc == 1 {
        for row in 0..rows {
            for col in 0..cols {
                let cptr = c.offset(rsc * row as isize + 1 * col as isize);
                (*cptr).scale_by(beta);
            }
        }
    } else {
        for col in 0..cols {
            for row in 0..rows {
                let cptr = c.offset(rsc * row as isize + csc * col as isize);
                (*cptr).scale_by(beta);
            }
        }
    }
}

unsafe fn zero_block<K: GemmKernel>(rows: usize,
                                    cols: usize,
                                    c: *mut K::Elem,
                                    rsc: isize,
                                    csc: isize) {

    if rsc == 1 {
        for col in 0..cols {
            for row in 0..rows {
                let cptr = c.offset(1 * row as isize + csc * col as isize);
                *cptr = K::Elem::zero();
            }
        }
    } else if csc == 1 {
        for row in 0..rows {
            for col in 0..cols {
                let cptr = c.offset(rsc * row as isize + 1 * col as isize);
                *cptr = K::Elem::zero();
            }
        }
    } else {
        for col in 0..cols {
            for row in 0..rows {
                let cptr = c.offset(rsc * row as isize + csc * col as isize);
                *cptr = K::Elem::zero();
            }
        }
    }
}

/// Allocate a vector of uninitialized data to be used for B~ and multiple A~ packing buffers.
///
/// + A~ needs be KC x MC x num_a
/// + B~ needs be KC x NC
/// but we can make them smaller if the matrix is smaller than this (just ensure
/// we have rounded up to a multiple of the kernel size).
///
/// Return packing vector, stride between of each app region, aligned pointer to start of first app, and aligned pointer to start of b
unsafe fn aligned_packing_vec<K>(m: usize, k: usize, n: usize, num_a: usize, align_to: usize) -> (Vec<K::Elem>, isize, *mut K::Elem, *mut K::Elem,)
    where K: GemmKernel
{
    let m = min(m, K::mc());
    let k = min(k, K::kc());
    let n = min(n, K::nc());
    // round up k, n to multiples of mr, nr
    // round up to multiple of kc
    
    debug_assert!(align_to % size_of::<K::Elem>() == 0);
    debug_assert!(align_to % align_of::<K::Elem>() == 0);
    
    let align_elems = align_to / size_of::<K::Elem>();

    let apack_size = k * round_up_to(m, K::mr());
    let bpack_size = k * round_up_to(n, K::nr());

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
        if cur_align != 0 {
            a_ptr = a_ptr.offset(((align_to - cur_align) / size_of::<K::Elem>()) as isize);
        }
    }
    
    let b_ptr = a_ptr.offset(((apack_size + align2)*num_a) as isize);
    
    (v, (apack_size + align2) as isize, a_ptr, b_ptr)
}



/// Pack matrix into `pack`
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + mr: kernel rows/columns that we round up to
/// + rsa: row stride
/// + csa: column stride
/// + zero: zero element to pad with
unsafe fn pack<T>(kc: usize,
                  mc: usize,
                  mr: usize,
                  pack: *mut T,
                  a: *const T,
                  rsa: isize,
                  csa: isize)
    where T: Element
{
    

    //if rsa == 1 || csa == 1 {
    //    vec_pack(kc,mc,mr,pack,a,rsa,csa);
    //} else {
        for ir in 0..mc / mr {
            let a = a.stride_offset(rsa, ir * mr);
            let pack = pack.offset((ir * mr * kc) as isize);

            
            let mut j = 0;
            unroll_by_4!(kc, {
                let mut i = 0;
                unroll_by_4!(mr, {
                    let pack = pack.offset((j*mr+i)as isize);
                    *pack = *a.stride_offset(rsa, i)
                            .stride_offset(csa, j);
                    i+=1;        
                });  
                j+=1;
            });    

        }
    //}


    let mut pack = pack.offset(((mc/mr)*mr*kc) as isize);

    let zero = <_>::zero();

    // Pad with zeros to multiple of kernel size (uneven mc)
    let rest = mc % mr;
    if rest > 0 {
        let row_offset = (mc / mr) * mr;
        for j in 0..kc {
            for i in 0..mr {
                if i < rest {
                    *pack = *a.stride_offset(rsa, i + row_offset)
                              .stride_offset(csa, j);
                } else {
                    *pack = zero;
                }
                pack.inc();
            }
        }
    }
}

/// Partial Pack matrix into `pack`
/// try and get llvm to generate simd 4x4 transpose code. Not currently working.
/// panics if neither csa==1 or rsa==1
#[allow(dead_code)]
#[allow(unused_assignments)]
unsafe fn vec_pack<T>(kc: usize,
                  mc: usize,
                  mr: usize,
                  pack: *mut T,
                  a: *const T,
                  rsa: isize,
                  csa: isize)
    where T: Element
{

    if csa == 1 {
        for ir in 0..mc / mr {
            let a = a.stride_offset(rsa, ir * mr);
            let pack = pack.offset((ir * mr * kc) as isize);

            for jo in 0..kc/4 {
               
                for io in 0..mr/4{
                    let mut pack = pack.offset(((jo*4)*mr+io*4)as isize);
                    let mut a = a.stride_offset(rsa, io*4).stride_offset(csa, jo*4);
                    let zero = <_>::zero();
                    let mut temp = [[zero;4];4];
                
                    loop4! (i, {
                        loop4! (j, {
                        
                            temp[j][i] = *a.offset(j as isize);
                        });
                        a = a.offset(rsa);
                    });

                    loop4! (j, {
                        loop4! (i, {
                            *pack.offset(i as isize) = temp[j][i];
                        });
                        pack = pack.offset(mr as isize);
                    });

                }
                
                for i in (mr/4)*4..mr{
                    let pack = pack.offset((jo*4*mr+i)as isize);
                    *pack = *a.stride_offset(rsa, i)
                            .stride_offset(csa, jo*4);
                }
            }

            for j in (kc/4)*4..kc {
                for i in 0..mr {
                    let pack = pack.offset((j*mr+i)as isize);
                    *pack = *a.stride_offset(rsa, i)
                            .stride_offset(csa, j);
                }         
            }
        }
    } else if rsa == 1 {
        for ir in 0..mc / mr {
            let a = a.stride_offset(rsa, ir * mr);
            let pack = pack.offset((ir * mr * kc) as isize);

            for jo in 0..kc/4 {

                for io in 0..mr/4{
                    let mut pack = pack.offset(((jo*4)*mr+io*4)as isize);
                    let mut a = a.stride_offset(rsa, io*4).stride_offset(csa, jo*4);

                    loop4! (_j, {
                        loop4! (i, {
                            *pack.offset(i) = *a.offset(i);
                        });
                        pack = pack.offset(mr as isize);
                        a = a.offset(csa);
                    });

                }
                for i in (mr/4)*4..mr{
                    let pack = pack.offset((jo*4*mr+i)as isize);
                    *pack = *a.stride_offset(rsa, i)
                            .stride_offset(csa, jo*4);
                }
            }

            for j in (kc/4)*4..kc {
                for i in 0..mr {
                    let pack = pack.offset((j*mr+i)as isize);
                    *pack = *a.stride_offset(rsa, i)
                            .stride_offset(csa, j);
                }         
            }
        }
    } else {
        unreachable!("either csa or rsa must equal 1");
    }

}



/// Call the GEMM kernel with a "masked" output C.
///
/// Simply redirect the MR by NR kernel output to the passed
/// in `mask_buf`, and copy the non masked region to the real
/// C.
///
/// + rows: rows of kernel unmasked
/// + cols: cols of kernel unmasked
#[inline(never)]
unsafe fn masked_kernel<T, K>(k: usize,
                              alpha: T,
                              a: *const T,
                              b: *const T,
                              c: *mut T,
                              rsc: isize,
                              csc: isize,
                              rows: usize,
                              cols: usize)
    where K: GemmKernel<Elem = T>,
          T: Element
{
    let mr = K::mr();
    let nr = K::nr();
    let mut ab_ = [T::zero(); MASK_SIZE];
    let mut ab = &mut ab_[0] as *mut T;
    K::kernel(k, T::one(), a, b, ab, 1, mr as isize);
    for j in 0..nr {
        for i in 0..mr {
            if i < rows && j < cols {
                let cptr = c.offset(rsc * i as isize + csc * j as isize);
                (*cptr).scaled_add(alpha, *ab);
            }
            ab.inc();
        }
    }
}

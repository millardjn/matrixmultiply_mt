use kernel::GemmKernel;
use archparam;
use unroll::Unroll;
use std::marker::PhantomData;

pub type T = f32;

pub struct Gemm<MR: Unroll<NR>, NR: Unroll<T>> {
	_phantom1: PhantomData<MR>,
	_phantom2: PhantomData<NR>,
}


impl<MR: Unroll<NR>, NR: Unroll<T>> GemmKernel for Gemm<MR, NR> {
    type Elem = T;

    #[inline(always)]
    fn align_to() -> usize {
        64
    }

    #[inline(always)]
    fn mr() -> usize {
        MR::val()
    }
    #[inline(always)]
    fn nr() -> usize {
        NR::val()
    }

    #[inline(always)]
    fn always_masked() -> bool {
        false
    }

    #[inline(always)]
    fn nc() -> usize {
        ((archparam::S_NC + NR::val() - 1) / NR::val()) * NR::val()
    }
    #[inline(always)]
    fn kc() -> usize {
        archparam::S_KC
    }
    #[inline(always)]
    fn mc() -> usize {
        ((archparam::S_MC + MR::val() - 1) / MR::val()) * MR::val()
    }

    #[inline(always)]
    unsafe fn kernel(k: usize,
                     alpha: T,
                     a: *const T,
                     b: *const T,
                     c: *mut T,
                     rsc: isize,
                     csc: isize) {
        Gemm::<MR, NR>::kernel(k, alpha, a, b, c, rsc, csc)
    }
}

impl<MR: Unroll<NR>, NR: Unroll<T>> Gemm<MR, NR> {
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

			let ab = Gemm::<MR, NR>::kernel_compute(k, alpha, a, b);
		
			Gemm::<MR, NR>::kernel_write(c, rsc, csc, &ab);		

	}


	/// Split out compute for better vectorisation
	#[inline(never)]
	unsafe fn kernel_compute(k: usize, alpha: T, a: *const T, b: *const T) -> MR{

		//let mut ab = *ab_;
		//loopMR!(i, loopNR!(j, ab[i][j] = 0.0)); // this removes the loads from stack, and xorps the registers

		let mut ab = MR::default();
		
		let mut a = a;
		let mut b = b;

		// Compute matrix multiplication into ab[i][j]
		// Due to llvm/MIR update a temporary array is no longer needed for vectorisation, and unroll doesnt ruin register allocation
		
		unroll_by_4!(k, {

			ab.unroll_self_mut(|ab, i|{
				ab[i].unroll_self_mut(|ab, j|{
					//ab[j] = at(a, i).mul_add(at(b, j) , ab[j]);
					ab[j] += at(a, i) * at(b, j);
				});	
			});	

			a = a.offset(MR::val() as isize);
			b = b.offset(NR::val() as isize);		

		});

		ab.unroll_self_mut(|ab, i|{
			ab[i].unroll_self_mut(|ab, j|{
				ab[j] *= alpha;
			});	
		});	

		ab
	}



	/// Choose writes to C in a cache/vectorisation friendly manner
	#[inline(always)]
	unsafe fn kernel_write(c: *mut T, rsc: isize, csc: isize, ab: & MR) {

		if rsc == 1 {
			ab.unroll_self(|ab, i|{
				ab[i].unroll_self(|ab, j|{
					*c.offset(1 * i as isize + csc * j as isize) += ab[j];
				});	
			});	
			
		} else if csc == 1 {
			ab.unroll_self(|ab, i|{
				ab[i].unroll_self(|ab, j|{
					*c.offset(rsc * i as isize + 1 * j as isize) += ab[j];
				});	
			});	
			
		} else {


			for i in 0..MR::val() {
				for j in 0..NR::val() {
					*c.offset(rsc * i as isize + csc * j as isize) += ab[i][j];
				}
			}
		}

	}


}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
	*ptr.offset(i as isize)
}

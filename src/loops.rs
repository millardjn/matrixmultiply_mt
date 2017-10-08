#[allow(unused)]
#[inline(always)]
pub fn partial_unroll<F: FnMut(usize, usize)>(n: usize, u:usize, f: &mut F){
	for i in 0..n/u {
		// let mut r = 0;
		// full_unroll(u, &mut |j|{f(j + u*i, r); r+=1;})
		partial_unroll_recurse(u, 0, i*u, f);
	}

	let mut r = 0;
	for i in (n/u)*u..n{
		f(i, r);
		r +=1;
	}
}

#[allow(unused)]
#[inline(always)]
fn partial_unroll_recurse<F: FnMut(usize, usize)>(n: usize, i: usize, offset: usize, f: &mut F) {
	f(i + offset, i);
	if i+1 < n {
		partial_unroll_recurse(n, i+1, offset, f);
	}
}

// #[inline(always)]
// pub fn full_unroll<F: FnMut(usize)>(n: usize, f: &mut F) {
// 	full_unroll_recurse(n, n-1, f);
// }

// #[inline(always)]
// pub fn full_unroll_recurse<F: FnMut(usize)>(n: usize, i: usize, f: &mut F) {
// 	if i > 0 {
// 		full_unroll_recurse(n, i-1, f);
// 	}
// 	f(i);
// }


#[inline(always)]
pub fn full_unroll<F: FnMut(usize)>(n: usize, f: &mut F) {
	//full_unroll_recurse(n, 0, f);
	for i in 0..n{
		f(i);
	}
}

// #[inline(always)]
// pub fn full_unroll_recurse<F: FnMut(usize)>(n: usize, i: usize, f: &mut F) {
// 	if n == i + 1 {
// 		f(i)
// 	} else {
// 		let h = i + (n - i)/2;
// 		full_unroll_recurse(h, i, f);
// 		full_unroll_recurse(n, h, f);
// 	}
// }


#[test]
fn test_partial() {
	let v1: Vec<usize> = (0..13).collect();
	let v2: Vec<usize> = (0..13).map(|i| i%5).collect();
	let mut v3 = vec![0; 13];
	let mut v4 = vec![0; 13];
	partial_unroll(13, 5, &mut |i, j|{v3[i] += i; v4[i] += j});

	assert_eq!(&v1, &v3);
	assert_eq!(&v2, &v4);
}

#[test]
fn test_full() {
	let v1: Vec<usize> = (0..7).collect();
	let mut v2 = vec![0; 7];
	full_unroll(7, &mut |i| v2[i] += i);

	assert_eq!(&v1, &v2);
}
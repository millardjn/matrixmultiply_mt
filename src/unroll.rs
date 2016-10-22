use std::ops::{Index,IndexMut};


pub trait Unroll<T: Default + Copy>:  'static + Default + Copy + Index<usize, Output=T> + IndexMut<usize, Output=T> { //

	fn val() -> usize;

	fn unroll<F: Fn(usize)>(f: F);
	fn unroll_self<F: Fn(&Self, usize)>(&self, f: F);
	fn unroll_self_mut<F: Fn(&mut Self, usize)>(&mut self, f: F);
	//fn ind(&self, index: usize) -> &mut T;
}




#[derive(Default, Copy, Clone)]
pub struct Unroll4<T: Default + Copy> {pub arr: [T; 4]}
impl<T: Default + Copy> Index<usize> for Unroll4<T>{
    type Output = T;
    fn index(&self, index: usize) -> &T { &self.arr[index] }
}
impl<T: Default + Copy> IndexMut<usize> for Unroll4<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output{ &mut self.arr[index] }
}
impl<T: 'static + Default + Copy> Unroll<T> for Unroll4<T> {
	fn val() -> usize {	4 }
	fn unroll<F: Fn(usize)>(f: F){
		f(0); f(1); f(2); f(3);
	}
	fn unroll_self<F: Fn(&Self, usize)>(&self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3);
	}
	fn unroll_self_mut<F: Fn(&mut Self, usize)>(&mut self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3);
	}
}


#[derive(Default, Copy, Clone)]
pub struct Unroll6<T: Default + Copy> {pub arr: [T; 6]}
impl<T: 'static + Default  + Copy> Unroll<T> for Unroll6<T> {
	fn val() -> usize {	6 }
	fn unroll<F: Fn(usize)>(f: F){
		f(0); f(1); f(2); f(3); f(4); f(5);
	}
	fn unroll_self<F: Fn(&Self, usize)>(&self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5);
	}
	fn unroll_self_mut<F: Fn(&mut Self, usize)>(&mut self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5);
	}
}
impl<T: Default + Copy> Index<usize> for Unroll6<T>{
    type Output = T;
    fn index(&self, index: usize) -> &T { &self.arr[index] }
}
impl<T: Default + Copy> IndexMut<usize> for Unroll6<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output{ &mut self.arr[index] }
}


#[derive(Default, Copy, Clone)]
pub struct Unroll8<T: Default + Copy> {pub arr: [T; 8]}
impl<T: 'static + Default + Copy> Unroll<T> for Unroll8<T> {
	fn val() -> usize {	8 }
	fn unroll<F: Fn(usize)>(f: F){
		f(0); f(1); f(2); f(3); f(4); f(5); f(6); f(7);
	}
	fn unroll_self<F: Fn(&Self, usize)>(&self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5); f(self, 6); f(self, 7);
	}
	fn unroll_self_mut<F: Fn(&mut Self, usize)>(&mut self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5); f(self, 6); f(self, 7);
	}
}
impl<T: Default + Copy> Index<usize> for Unroll8<T>{
    type Output = T;
    fn index(&self, index: usize) -> &T { &self.arr[index] }
}
impl<T: Default + Copy> IndexMut<usize> for Unroll8<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output{ &mut self.arr[index] }
}


#[derive(Default, Copy, Clone)]
pub struct Unroll16<T: Default + Copy> {pub arr: [T; 16]}
impl<T: 'static + Default + Copy> Unroll<T> for Unroll16<T> {
	fn val() -> usize {	16 }
	fn unroll<F: Fn(usize)>(f: F){
		f(0); f(1); f(2); f(3); f(4); f(5); f(6); f(7);
		f(8); f(9); f(10); f(11); f(12); f(13); f(14); f(15);
	}
	fn unroll_self<F: Fn(&Self, usize)>(&self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5); f(self, 6); f(self, 7);
		f(self, 8); f(self, 9); f(self, 10); f(self, 11); f(self, 12); f(self, 13); f(self, 14); f(self, 15);
	}
	fn unroll_self_mut<F: Fn(&mut Self, usize)>(&mut self, f: F){
		f(self, 0); f(self, 1); f(self, 2); f(self, 3); f(self, 4); f(self, 5); f(self, 6); f(self, 7);
		f(self, 8); f(self, 9); f(self, 10); f(self, 11); f(self, 12); f(self, 13); f(self, 14); f(self, 15);
	}
}
impl<T: Default + Copy> Index<usize> for Unroll16<T>{
    type Output = T;
    fn index(&self, index: usize) -> &T { &self.arr[index] }
}
impl<T: Default + Copy> IndexMut<usize> for Unroll16<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output{ &mut self.arr[index] }
}


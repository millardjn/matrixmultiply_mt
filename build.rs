use std::env;

fn main() {
	println!("cargo:rerun-if-changed=build.rs");

	let mut arch_flag_set = false;
	if let Ok(flags) = env::var("MATMULFLAGS") {

		let user_flags = flags.split(",").map(|s| s.trim()).collect::<Vec<_>>();

		let arch_flags = &[
				"arch_generic4x4",
				"arch_generic4x4fma",
				"arch_penryn",
				"arch_sandybridge",
				"arch_haswell",
			];

		let other_flags = &[
				"ftz_daz",
				"prefetch",
				"no_multithreading",
			];
		
		for user_flag in user_flags.iter() {
			if arch_flags.iter().any(|flag| flag == user_flag) {
				println!("cargo:rustc-cfg={}", user_flag);
				arch_flag_set = true;
			} else if other_flags.iter().any(|flag| flag == user_flag) {
				println!("cargo:rustc-cfg={}", user_flag);
			} else {
				panic!("matrixmultiply_mt: Environment variable MATMULFLAGS contained unrecognised flag: {}", user_flag);
			}
		}
	} else {
		println!("matrixmultiply_mt: Environment variable MATMULFLAGS was not set");
	}

	// if no architechture specified, take a guess
	if !arch_flag_set{
		if let Ok(features) = env::var("CARGO_CFG_TARGET_FEATURE"){
			println!("matrixmultiply_mt: Guessing CPU architecture from features: {}", features);
			if features.split(",").map(|s| s.trim()).any(|feat| feat == "avx2") {
				println!("cargo:rustc-cfg=arch_haswell");
			} else if features.split(",").map(|s| s.trim()).any(|feat| feat == "avx") {
				println!("cargo:rustc-cfg=arch_sandybridge");
			} else if features.split(",").map(|s| s.trim()).any(|feat| feat == "64bit-mode")
				&& features.split(",").map(|s| s.trim()).any(|feat| feat == "sse") {
				println!("cargo:rustc-cfg=arch_penryn");
			} else if features.split(",").map(|s| s.trim()).any(|feat| feat == "fma" || feat == "fma4") {
				println!("cargo:rustc-cfg=generic4x4fma");
			}
		} else {
			println!("matrixmultiply_mt: Cannot guess CPU architecture as CARGO_CFG_TARGET_FEATURE is not available, use environment variable MATMULFLAGS to set arch");
		}
	}

}
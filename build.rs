use std::env;

fn main() {
	println!("cargo:rerun-if-changed=build.rs");

	if let Ok(flags) = env::var("MATMULFLAGS") {

		let user_flags = flags.split(",").map(|s| s.trim()).collect::<Vec<_>>();

		let cfg_flags = &[
				"arch_generic4x4",
				"arch_generic4x4fma",
				"arch_penryn",
				"arch_sandybridge",
				"arch_haswell",
				"ftz_daz",
			];
		
		for user_flag in user_flags.iter() {
			if cfg_flags.iter().any(|flag| flag == user_flag) {
				println!("cargo:rustc-cfg={}", user_flag);
			} else {
				panic!("Environment variable MATMULFLAGS contained unrecognised flag: {}", user_flag);
			}
		}
	} else {
		println!("Environment variable MATMULFLAGS was not set");
	}
}
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    
    // Force cc crate to use gcc 13 for nvcc host compilation, avoiding unsupported gcc 14 error
    env::set_var("CXX", "g++-13");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-arch=sm_61") // Target architecture for GTX 1080 Ti (Pascal)
        .flag("-O3")
        .flag("-use_fast_math")
        .file("cuda/bindings.cu")
        .file("cuda/physics.cu")
        .compile("libgenesis_cuda.a");

    // Automatically link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
}

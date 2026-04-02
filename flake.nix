{
  description = "LLM Bench - Compare tok/s between llama-cpp-rs and mistral-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        # Pin Rust to match your installed version
        rustToolchain = pkgs.rust-bin.stable."1.94.1".default;

        # CUDA packages (x86_64-linux only)
        cudaPkgs =
          if system == "x86_64-linux" then
            with pkgs.cudaPackages;
            [
              cudatoolkit
              cuda_nvcc
            ]
          else
            [ ];

        buildInputs =
          with pkgs;
          [
            llvmPackages.libclang
            gcc
          ]
          ++ cudaPkgs;

        nativeBuildInputs = with pkgs; [
          rustToolchain
          llvmPackages.llvm
          cmake
          ninja
          pkg-config
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "llm-bench";

          buildInputs = buildInputs;
          nativeBuildInputs = nativeBuildInputs;

          shellHook = ''
            # bindgen needs libclang
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            export LD_LIBRARY_PATH="${pkgs.llvmPackages.libclang.lib}/lib:$LD_LIBRARY_PATH"

            # CUDA env (only set if cudaPackages are present)
            ${pkgs.lib.optionalString (cudaPkgs != [ ]) ''
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
              export CUDA_HOME="$CUDA_PATH"
              export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"

              export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH"
              # Static linker needs LIBRARY_PATH to find cudart_static.a
              export LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:$LIBRARY_PATH"

              # Nix CUDA merged package ships cudart_static.a and culibos.a
              # but NOT cublas_static.a / cublasLt_static.a (only .so).
              # llama-cpp-sys-2 build.rs hardcodes static linking on Linux,
              # so we create dummy .a archives and dynamically link the real libs.
              WORKAROUND_DIR=$(mktemp -d)
              touch "$WORKAROUND_DIR/dummy.c"
              ${pkgs.gcc}/bin/gcc -c "$WORKAROUND_DIR/dummy.c" -o "$WORKAROUND_DIR/dummy.o"
              for lib in cublas_static cublasLt_static; do
                ${pkgs.gcc}/bin/ar rcs "$WORKAROUND_DIR/lib''${lib}.a" "$WORKAROUND_DIR/dummy.o"
              done
              trap "rm -rf $WORKAROUND_DIR" EXIT

              # Nix merged CUDA puts libs in lib/ not lib64/ —
              # find_cuda_helper crate may look in the wrong place,
              # so explicitly tell the Rust linker where to search.
              # Include the workaround dir for dummy .a stubs and
              # dynamic links for the real CUDA shared libs.
              export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L ${pkgs.cudaPackages.cudatoolkit}/lib -L $WORKAROUND_DIR -l cublas -l cublasLt"
            ''}

            # Use gcc wrapper (it wraps clang + has correct glibc include paths)
            export CC="${pkgs.gcc}/bin/gcc"
            export CXX="${pkgs.gcc}/bin/g++"

            # bindgen uses libclang directly — give it glibc headers
            export BINDGEN_EXTRA_CLANG_ARGS="-I${pkgs.stdenv.cc.cc.lib.dev or pkgs.glibc.dev}/include"
            # Extract glibc dev include path from gcc wrapper's own specs
            GLIBC_INC=$(gcc -xc -E -Wp,-v - < /dev/null 2>&1 | grep "^ " | tr -d ' ' | grep glibc | head -1)
            if [ -n "$GLIBC_INC" ]; then
              export BINDGEN_EXTRA_CLANG_ARGS="-I$GLIBC_INC"
            fi

            # Avoid cmake/ninja rebuilds from nix store differences
            export CMAKE_GENERATOR="Ninja"
            export NINJA_STATUS="[%u/%r/%f] "

            echo "🔧  llm-bench dev shell"
            echo "    rust: $(rustc --version)"
            echo "    gcc: $(gcc --version | head -1)"
            ${pkgs.lib.optionalString (cudaPkgs != [ ]) ''
              echo "    cuda: $CUDA_PATH"
            ''}
            echo ""
            echo "    cargo build --features cuda --release"
          '';
        };
      }
    );
}

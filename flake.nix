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
            clang
            cmake
            pkg-config
            gcc
          ]
          ++ cudaPkgs;

        nativeBuildInputs = with pkgs; [
          rustToolchain
          llvmPackages.llvm
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
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang}/lib"

            # CUDA env (only set if cudaPackages are present)
            ${pkgs.lib.optionalString (cudaPkgs != [ ]) ''
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
              export CUDA_HOME="$CUDA_PATH"
              export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib64:$LD_LIBRARY_PATH"
            ''}

            # Cargo uses clang for C++ compilation via bindgen
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"

            # Avoid cmake/ninja rebuilds from nix store differences
            export CMAKE_GENERATOR="Ninja"
            export NINJA_STATUS="[%u/%r/%f] "

            echo "🔧  llm-bench dev shell"
            echo "    rust: $(rustc --version)"
            echo "    clang: $(clang --version | head -1)"
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

with import <nixpkgs> {};

pkgs.mkShell {
	nativeBuildInputs = with pkgs; [python3Packages.torchWithRocm python3Packages.accelerate python3Packages.transformers python3Packages.openai zig zls llama-cpp-vulkan];
}

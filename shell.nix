with import <nixpkgs> {};

pkgs.mkShell {
	nativeBuildInputs = with pkgs; [python3Packages.torchWithRocm python3Packages.accelerate python3Packages.transformers python3Packages.openai python3Packages.onnxscript python3Packages.onnxruntime python3Packages.numpy zig zls llama-cpp-vulkan onnxruntime];
}

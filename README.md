# Sudoku solver

The idea is to create a sudoku solver with OCR integration. The core solver written in c++ for maximum speed.

The program is structured in 4 steps:
1. Sudoku detection
2. Sudoku recongnition (cells)
3. Digits recognition
4. Sudoku solver

## Installation

For CPP (vendor-based dependencies; no Conan required):
1. install opencv and onnxruntime via homebrew or python fetch_deps.py
2. build.sh macos
3. ./build_macos/main ../images/cover.jpg

For python (digit recongnition model):
1. Install "uv" package manager
2. Run: `uv sync`

## Run

Example:

```bash
./build/main ./models/models/resnet18_svhn_8bit.onnx ./images/cover.jpg
```



<!-- 
Mnist models are too weak. Other models to try: SVHN, Char74K, EMNIST, Tesseract (digit only mode), heavy models...

- https://huggingface.co/edadaltocg/resnet18_svhn/tree/main
- https://huggingface.co/qualcomm/EasyOCR/tree/main 
-->





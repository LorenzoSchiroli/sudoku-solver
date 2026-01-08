# Sudoku solver

The idea is to create a sudoku solver with OCR integration. The core solver written in c++ for maximum speed.

The program is structured in 4 steps:
1. Sudoku detection
2. Sudoku recongnition (cells)
3. Digits recognition
4. Sudoku solver

## Installation

For CPP:
1. Install "CMake"
2. Install "Conan" package manager
3. Run: `./build.sh`

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





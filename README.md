# Sudoku solver

Sudoku solver module and app with vision capabilities. The core vision and solver are written in c++ for maximum speed.

The vision core is structured in 4 steps:
1. Sudoku detection
2. Sudoku recongnition (cells)
3. Digits recognition
4. Sudoku solver

There is also the mobile app and a ml pipeline to generate the digit recognition model.

## Installation

For the vision core:
1. install opencv and onnxruntime via homebrew or python fetch_deps.py
2. sh ./build.sh macos
3. ./build_macos/main ../images/cover.jpg

For the ml model pipeline (digit recognition):
1. Install "uv" package manager
2. Run: `uv sync`

For the mobile app:
1. move models/resnet18_svhn_int8.onnx in app-mobile/assets/resnet18_svhn_int8.onnx
2. run `flutter run` for debug (`flutter run --profile` for debugging speed)
3. run `flutter build apk --release --target-platform android-arm64` for release


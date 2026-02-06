# Sudoku solver

A sudoku solver mobile app with vision capabilities. The computer vision and solver algorithm core are written in c++ and designed for maximum speed and efficiency. The mobile app is written in Flutter.

The computer vision and solver algorithm core is structured in 4 steps:
1. **Sudoku detection**: detection of the board of the sudoku. The module search for the biggest square and crop it correcting the perspective.
2. **Sudoku recongnition**: recognizing the sudoku cells. The module find the filled cells and crops the numbers inside them.
3. **Digits recognition**: recognizing the digits (using a deep learning model) given the cropped images.
4. **Sudoku solver**: algorithm to solve the sudoku. 
All these steps combined takes a total of 140ms average on mobile CPU (Google Tensor 1).

The Android app is made with Flutter, while the deep learning model for digit recongnition is made with Pytorch and Huggigface.

The the deep learning degit recongnition model is deployed with optimizations:
- CNN architecture (efficient for edge devices)
- converted to onnx format
- 8bit static quatization
- batch processing
More specifically, it's a resnet18 trained on SVHN dataset.

The sudoku solver algorithm is implemented in an efficient way too. Here some optimizations:
- Bitmasks: Rows, cols, and boxes use 16-bit integers to track used numbers.
- MRV Heuristic: Always branches on the cell with the fewest valid options first.
- Lookup Tables: Pre-computed indices to avoid division/modulo operations in the hot loop.
- Cache Locality: Uses a flat std::array.

## Installation

1. Generate the digit recognition model (resnet18_svhn_int8.onnx):
    1. `cd ml-pipeline`
    2. Install UV package manager
    3. Run: `uv sync`
    4. Run: `python digit_recognition_model.py`

2. Download dependencies for the c++ vision core:
    1. `cd core-vision`
    2. Run: `python fetch_deps.py`

3. Create the mobile app (android):
    1. `cd app-mobile`
    2. Install Flutter
    3. Move models/resnet18_svhn_int8.onnx in app-mobile/assets/resnet18_svhn_int8.onnx
    4. run `flutter build apk --release --target-platform android-arm64` for release

## Debugging

Vision core:
- (macos) install dependencies with homebrew (opencv and onnxruntime)
- sh ./build.sh macos
- ./build_macos/main ../images/cover.jpg

Flutter app:
- run `flutter run` for debug (`flutter run --profile` for debugging speed performances)


<!--
**Improvement ideas**

Vision core:
- autodetect: stream to autodetect the sudoku without taking a photo (idea: keeping it simple we can just detect a square like now with precise size range and accept it immediately)
- discriminator model to detect hand-written vs printed digits (this way the app would work even after a person completed it)

App side:
- add a show/hide button to show/hide the solution
- add a history for the detected boards
- run cpp module on a separate thread

Add to play store?:
- add advertisements or freemimum feature
- build visibility
-->
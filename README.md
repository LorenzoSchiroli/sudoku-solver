# Sudoku Recognition and Solver
### Low-latency on-device OCR and constraint-solving pipeline for mobile devices

A high-performance computer vision and ML inference on edge hardware to detect, recognize, and solve sudoku puzzles in real-time. Built with a focus on speed and efficiency, combining advanced C++ algorithms with a modern Flutter frontend.

https://github.com/user-attachments/assets/b787e070-1abc-4f62-ac0a-6e9cc3e9d37d

**Key Features:**
- **Rapid Processing**: Complete vision-to-solution pipeline completes in ~110ms on mobile hardware (Google Tensor G1)
- **Deep Learning Integration**: Optimized ResNet18 model for digit recognition deployed on edge devices
- **Efficient Solving**: Constraint-propagation algorithm with advanced optimization techniques

## Table of Contents

- [Architecture & Implementation](#architecture--implementation)
  - [Computer Vision Pipeline](#computer-vision-pipeline)
  - [Model Optimization Strategy](#model-optimization-strategy)
  - [Solving Algorithm](#solving-algorithm)
  - [Design tradeoffs](#design-tradeoffs)
  - [Strength and limitations](#strength-and-limitations)
- [Tech Stack](#tech-stack)
- [Installation & Build](#installation--build)
- [Development & Debugging](#development--debugging)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Architecture & Implementation

### Computer Vision Pipeline

The solution processes captured images through a sophisticated 4-stage pipeline:

1. **Sudoku Detection**
   - Identifies the largest square region in the frame and extracts it with perspective correction
   - Handles real-world challenges like skewed camera angles and varied lighting

2. **Cell Recognition**
   - Localizes individual cells within the detected grid
   - Isolates digit regions from empty cells using automated segmentation

3. **Digit Classification**
   - Employs a deep learning model (ResNet18 trained on SVHN dataset) for robust digit recognition
   - Deployed with multiple optimizations for mobile inference

4. **Sudoku Solving**
   - Executes constraint-satisfaction algorithm to compute the solution
   - Optimized for minimal latency

### Model Optimization Strategy

The digit recognition model is optimized across multiple dimensions:

| Optimization | Benefit |
|---|---|
| **CNN Architecture** | Lightweight design suited for edge device inference |
| **ONNX Format** | Framework-agnostic, optimized runtime execution |
| **8-bit Quantization** | 4x reduced model size with minimal accuracy loss |
| **Batch Processing** | Vectorized operations for efficient cell processing |

### Solving Algorithm

The sudoku solver employs several key optimizations for competitive performance:

- **Bitmask Constraints**: 16-bit integers track occupied cells per row, column, and 3×3 box
- **MRV Heuristic**: Prioritizes cells with minimum remaining values to reduce search space
- **Lookup Tables**: Pre-computed indices eliminate costly division/modulo operations in hot loops
- **Cache Optimization**: Flat `std::array` structure maximizes CPU cache locality

### Design tradeoffs

- CNN for digit recognition: Classical template-based OCR approaches were insufficient under perspective distortion and font variability, while full OCR systems were unnecessarily heavy. A lightweight CNN trained for single-digit classification provided the best accuracy–latency tradeoff.
- Classical computer vision for grid detection: Board and cell detection are implemented using geometric and morphological operations rather than neural networks, as they are faster, deterministic, and sufficiently robust for this task.
- SVHN instead of MNIST: MNIST-trained models showed poor generalization to real-world printed digits, while SVHN provided significantly better robustness to font style, scale, and noise.
- ResNet18: Chosen as the smallest architecture with stable convergence on SVHN while remaining quantization-friendly.
- ONNX Runtime: Selected over alternatives for easier C++ integration and cross-platform consistency.

### Strength and limitations

These design decisions result in a system with the following strengths and known limitations.

#### Strengths

The system was validated against a variety of real-world conditions:
- Robust to strong perspective distortion and oblique viewing angles (excluding curved surfaces)
- Supports multiple sudoku boards within a single frame
- Works with both printed boards and digital displays (moiré artifacts mitigated)
- Handles varied fonts, digit sizes, and board color schemes

#### Limitations

- Performance degrades under very low-light conditions (mitigated via optional camera flash)
- Performance degrades on scribbled or dirty boards
- Model is trained exclusively on printed digits; handwritten digits are out of scope
- Board detection assumes the presence of a clear outer square border


## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Mobile App** | Flutter, Dart | Cross-platform UI and camera integration |
| **Vision Core** | C++23, OpenCV, ONNX Runtime | Performance-critical image processing and inference |
| **ML Model** | PyTorch, Hugging Face | Conversion of digit recognition model |
| **Build System** | CMake, Gradle | Multi-platform compilation and packaging |

## Installation & Build

### Prerequisites
- Flutter SDK
- C++ compiler (clang/gcc)
- CMake 4.2+
- Python 3.12+ with UV package manager

### Setup Instructions

**1. ML Pipeline** (generates ResNet18_SVHN_INT8 model)
```bash
cd ml-pipeline
uv sync
python digit_recognition_model.py
```

**2. Vision Core** (downloads and builds C++ dependencies)
```bash
cd core-vision
python fetch_deps.py
```

For debugging on macos:
```bash
sh ./build.sh macos
# example
./build_macos/main ../images/cover.jpg
```

**3. Mobile App** (Android release build)
```bash
cd app-mobile
# Ensure model is in place
cp ../models/resnet18_svhn_int8.onnx ./assets/
# Build optimized release APK
flutter build apk --release --target-platform android-arm64
```

For debugging:
```bash
flutter run
```

## Development & Debugging


Potential improvements and roadmap items:

**Vision Core**
- Continuous stream detection: Real-time grid detection without requiring explicit photo capture <!-- idea: keep square detection and add area range to catch it >
- Handwritten digit recognition with discriminator capability: Multi-model approach to handle both printed and handwritten numbers, with the capability to distinbuish the two <!-- idea: fiewer resnet layers, build dataset with handwritten mnist/emnist + digital synthetic, add head to discriminate >

**Mobile App**
- Solution visualization toggle: Show/hide computed solution
- Detection history: Browse previously captured and solved puzzles
- Async processing: Execute C++ solver on background thread to maintain responsive UI

## Contributing

This project demonstrates practical application of computer vision, deep learning optimization, and performance engineering. Contributions and suggestions for improvements are welcome.

## License

See [LICENSE](LICENSE) file for details.



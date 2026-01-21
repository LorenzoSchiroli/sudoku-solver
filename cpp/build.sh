#!/bin/bash
set -e

PLATFORM=$1 # [macos|ios|android]
BUILD_TYPE="Release"
[[ "$2" == "--debug" ]] && BUILD_TYPE="Debug"

BUILD_DIR="build_$PLATFORM"

[ -d $BUILD_DIR ] && rm -r $BUILD_DIR

# Basic CMake call
FLAGS=""
if [[ "$PLATFORM" == "android" ]]; then
    FLAGS="-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a"
elif [[ "$PLATFORM" == "ios" ]]; then
    FLAGS="-DCMAKE_SYSTEM_NAME=iOS -G Xcode"
elif [[ "$PLATFORM" == "macos" ]]; then
    ONNXRUNTIME_DIR=$(ls -d $(brew --prefix)/Cellar/onnxruntime/* | head -n 1)
    FLAGS="-DCMAKE_PREFIX_PATH=$ONNXRUNTIME_DIR"
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $FLAGS
cmake --build "$BUILD_DIR" --config $BUILD_TYPE -j 4
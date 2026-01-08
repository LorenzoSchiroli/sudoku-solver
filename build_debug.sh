[ -d build ] && rm -r build

# 1. Install Conan dependencies for Debug
conan install . --build=missing -s build_type=Debug

# 2. Configure CMake for Debug
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=build/Debug/generators/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Debug

# 3. Build
cmake --build build
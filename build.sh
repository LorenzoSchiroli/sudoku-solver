# [ -d build ] && rm -r build
# mkdir build
# cd build
# cmake .. -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# make

[ -d build ] && rm -r build

conan install . \
  --build=missing

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

cmake --build build


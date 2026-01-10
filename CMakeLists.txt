cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(MyOpenCVProject CXX)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# Build each module as a static library (library-style: header + source)
add_library(sudoku_solver STATIC src/sudoku_solver.cpp)
add_library(sudoku_detection STATIC src/sudoku_detection.cpp)
add_library(grid_extraction STATIC src/grid_extraction.cpp)
add_library(digit_recognition STATIC src/digit_recognition.cpp)

# Link 3rd-party libs to the libraries that need them
target_link_libraries(sudoku_detection PRIVATE opencv::opencv)
target_link_libraries(grid_extraction PRIVATE opencv::opencv)
target_link_libraries(digit_recognition PRIVATE opencv::opencv onnxruntime::onnxruntime)

target_include_directories(sudoku_solver PUBLIC src)
target_include_directories(sudoku_detection PUBLIC src)
target_include_directories(grid_extraction PUBLIC src)
target_include_directories(digit_recognition PUBLIC src)

# Integration executable
add_executable(main app/main.cpp)
target_link_libraries(main PRIVATE sudoku_detection grid_extraction digit_recognition sudoku_solver opencv::opencv onnxruntime::onnxruntime)

# Per-module test exes (contain the previous mains)
add_executable(test_sudoku_solver tests/test_sudoku_solver.cpp)
target_link_libraries(test_sudoku_solver PRIVATE sudoku_solver)

add_executable(test_sudoku_detection tests/test_sudoku_detection.cpp)
target_link_libraries(test_sudoku_detection PRIVATE sudoku_detection opencv::opencv)

add_executable(test_grid_extraction tests/test_grid_extraction.cpp)
target_link_libraries(test_grid_extraction PRIVATE grid_extraction opencv::opencv)

add_executable(test_digit_recognition tests/test_digit_recognition.cpp)
target_link_libraries(test_digit_recognition PRIVATE digit_recognition onnxruntime::onnxruntime opencv::opencv)


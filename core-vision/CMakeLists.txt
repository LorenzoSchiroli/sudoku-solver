cmake_minimum_required(VERSION 3.16)
project(MyOpenCVProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(Dependencies.cmake)

# ---- Libraries ----

add_library(sudoku_solver STATIC src/sudoku_solver.cpp)
target_include_directories(sudoku_solver PUBLIC src)

add_library(sudoku_detection STATIC src/sudoku_detection.cpp)
target_link_libraries(sudoku_detection PRIVATE ${OpenCV_LIBS})

add_library(grid_extraction STATIC src/grid_extraction.cpp)
target_link_libraries(grid_extraction PRIVATE ${OpenCV_LIBS})

add_library(digit_recognition STATIC src/digit_recognition.cpp)
target_link_libraries(digit_recognition
    PUBLIC onnxruntime::onnxruntime
    PRIVATE ${OpenCV_LIBS}
)

add_library(sudoku_main STATIC src/sudoku_main.cpp)
target_include_directories(sudoku_main PUBLIC src)
target_link_libraries(sudoku_main PRIVATE
    sudoku_detection
    grid_extraction
    digit_recognition
    sudoku_solver
    ${OpenCV_LIBS}
    PUBLIC onnxruntime::onnxruntime
)

# ---- CLI (dev only) ----

if(NOT ANDROID AND NOT IOS)
    add_executable(main cli/main.cpp)
    target_link_libraries(main PRIVATE sudoku_main ${OpenCV_LIBS})
endif()

# ---- Flutter bridge ----

# find_library(log-lib log)
add_library(sudoku_bridge SHARED src/flutter_bridge.cpp)
target_include_directories(sudoku_bridge PUBLIC src)
target_link_libraries(sudoku_bridge PRIVATE sudoku_main ${OpenCV_LIBS}) #${log-lib})

set_target_properties(sudoku_bridge PROPERTIES
    OUTPUT_NAME "sudoku_bridge"
)

# # ---- Tests ----

# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT ANDROID AND NOT IOS)
#     enable_testing()
#     add_executable(test_sudoku_solver tests/test_sudoku_solver.cpp)
#     target_link_libraries(test_sudoku_solver PRIVATE sudoku_solver)
# endif()



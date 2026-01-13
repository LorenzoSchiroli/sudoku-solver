#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Simple visibility macro for exported functions
#if defined(_WIN32)
  #if defined(SUDOKU_BRIDGE_IMPLEMENTATION)
    #define SUDOKU_BRIDGE_API __declspec(dllexport)
  #else
    #define SUDOKU_BRIDGE_API __declspec(dllimport)
  #endif
#else
  #define SUDOKU_BRIDGE_API __attribute__((visibility("default")))
#endif

// Status values mirror the C++ enum class SudokuStatus
enum SudokuStatusC : int32_t {
    SudokuStatus_GridNotFound = 0,
    SudokuStatus_NotSolved   = 1,
    SudokuStatus_Solved      = 2
};

// C representation of a cell (row-major ordering expected)
typedef struct {
    int32_t number; // 0..9
    int32_t mask;   // 0 or 1
} CellC;

// Flattened 9x9 grid in row-major order
typedef struct {
    int32_t status; // SudokuStatusC
    // 81 cells (9 rows * 9 cols)
    CellC cells[81];
} SudokuResultC;

// Lifecycle for processor (opaque pointer representing a SudokuMain instance)
// model_path may be nullptr to use the default model path.
SUDOKU_BRIDGE_API void* sudoku_create_processor(const char* model_path);
SUDOKU_BRIDGE_API void sudoku_destroy_processor(void* processor);

// Process image from bytes (e.g., JPEG/PNG content). Returns a pointer to an
// allocated SudokuResultC which must be freed with sudoku_free_result.
// Returns nullptr on fatal allocation error.
SUDOKU_BRIDGE_API SudokuResultC* sudoku_process_image_bytes(void* processor, const uint8_t* data, size_t length);

// Convenience: process image from file path
SUDOKU_BRIDGE_API SudokuResultC* sudoku_process_image_file(void* processor, const char* path);

// Free result returned by any of the processing functions
SUDOKU_BRIDGE_API void sudoku_free_result(SudokuResultC* result);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma once

#include <cstddef>
#include <cstdint>

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
  SudokuStatus_NotSolved = 1,
  SudokuStatus_Solved = 2,
  SudokuStatus_InitializationError = 3
};

typedef struct {
  int32_t number; // 0..9
  int32_t mask;   // 0 or 1
} CellC;

// Flattened 9x9 grid in row-major order
typedef struct {
  int32_t status;
  // 81 cells (9 rows * 9 cols)
  CellC cells[81];
} SudokuResultC;

// `model_path` may be nullptr to use the library default model path.
/**
 * Create a new processor instance.
 *
 * @param model_path Optional path to the digit-recognition model.
 * @return Opaque pointer to the processor or nullptr on failure.
 */
SUDOKU_BRIDGE_API void *sudoku_create_processor(const char *model_path);

/**
 * Destroy a processor previously returned by `sudoku_create_processor`.
 */
SUDOKU_BRIDGE_API void sudoku_destroy_processor(void *processor);

/**
 * Process an in-memory image buffer (e.g., PNG/JPEG bytes).
 *
 * The function returns a newly allocated `SudokuResultC*` which the caller
 * must free with `sudoku_free_result`. On fatal errors this returns an
 * error-encoded result instead of throwing.
 */
SUDOKU_BRIDGE_API SudokuResultC *
sudoku_process_image_bytes(void *processor, const uint8_t *data, size_t length);

/**
 * Free a `SudokuResultC*` returned by the processing functions.
 */
SUDOKU_BRIDGE_API void sudoku_free_result(SudokuResultC *result);

#ifdef __cplusplus
}
#endif

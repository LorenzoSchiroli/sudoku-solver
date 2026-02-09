#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "digit_recognition.hpp"

enum class SudokuStatus { GridNotFound, NotSolved, Solved };

// Cell: carries the digit value and a mask indicating whether the value
// was detected from the image (mask==1) or filled by the solver (mask==0).
struct Cell {
    int number; // digit value (0 for empty)
    int mask;   // 1 = originally detected/predicted from image, 0 = filled by solver
};

struct SudokuResult {
    std::vector<std::vector<Cell>> grid;
    SudokuStatus status;
};

/**
 * High-level orchestrator that combines board detection, grid extraction,
 * digit recognition and solving. Use `sudoku_img2grid` to process an image
 * and obtain a `SudokuResult` describing the solved or partially-solved grid.
 */
class SudokuMain {
public:
    /**
     * Construct the processor and load the digit-recognition model.
     * If `modelPath` is omitted, a sensible default bundled with the app is used.
     */
    explicit SudokuMain(const std::string& modelPath = "resnet18_svhn_int8.onnx");

    /**
     * Process a color image and return the detected/solved Sudoku grid.
     *
     * @param originalImage Input BGR image containing a single Sudoku board.
     * @return SudokuResult containing the grid and an operation status.
     */
    SudokuResult sudoku_img2grid(const cv::Mat& originalImage);

private:
    DigitRecognizer recognizer;
};

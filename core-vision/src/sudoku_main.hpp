#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "digit_recognition.hpp"

enum class SudokuStatus { GridNotFound, NotSolved, Solved };

// Add Cell type used to carry (number, mask) per cell
struct Cell {
	int number; // digit value (0 for empty)
	int mask;   // 1 = originally detected/predicted from image, 0 = filled by solver
};

struct SudokuResult {
    std::vector<std::vector<Cell>> grid;
    SudokuStatus status;
};

class SudokuMain {
public:
    // Construct and load the digit recognition model (default path provided)
    explicit SudokuMain(const std::string& modelPath = "resnet18_svhn_int8.onnx");

    // Process an image and return a SudokuResult
    SudokuResult sudoku_img2grid(const cv::Mat& originalImage);

private:
    DigitRecognizer recognizer;
};

// // Backwards-compatible free function that uses a single static instance
// SudokuResult sudoku_img2grid(const cv::Mat& originalImage);

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // added for timing

#include "sudoku_detection.hpp"
#include "grid_extraction.hpp"
#include "digit_recognition.hpp"
#include "sudoku_solver.hpp"
#include "sudoku_main.hpp"

/**
 * Construct SudokuMain and load the digit-recognition model.
 */
SudokuMain::SudokuMain(const std::string& modelPath)
    : recognizer(modelPath) {}

/**
 * Process a color image and return the detected/solved Sudoku grid.
 * This function orchestrates detection, grid extraction, recognition and solving.
 */
SudokuResult SudokuMain::sudoku_img2grid(const cv::Mat& originalImage) {

    std::vector<std::vector<Cell>> sudokuGrid(9, std::vector<Cell>(9, {0, 0}));

    // --- 1. Sudoku Detection ---
    // auto t1_start = std::chrono::high_resolution_clock::now();
    std::optional<cv::Mat> boardCrop = detectSudokuBoard(originalImage);
    // auto t1_end = std::chrono::high_resolution_clock::now();
    // auto t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count();
    // std::cout << "Step 1 (Sudoku Detection) took: " << t1_ms << " ms" << std::endl;

    // Check Detection
    if (!boardCrop.has_value()) {
        return SudokuResult{ sudokuGrid, SudokuStatus::GridNotFound };
    }

    // --- 2. Grid Extraction ---
    // auto t2_start = std::chrono::high_resolution_clock::now();
    // Returns 9x9 grid of cell images (empty cells are handled internally by your function)
    std::vector<std::vector<cv::Mat>> cellCrops = analyze_sudoku_board(*boardCrop);
    // auto t2_end = std::chrono::high_resolution_clock::now();
    // auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count();
    // std::cout << "Step 2 (Grid Extraction) took: " << t2_ms << " ms" << std::endl;

    // --- 3. Digit Recognition ---
    // auto t3_start = std::chrono::high_resolution_clock::now();
    // DigitRecognizer recognizer("resnet18_svhn_int8.onnx");
    std::vector<std::vector<int>> puzzleGrid = recognizer.predictGrid(cellCrops);
    // auto t3_end = std::chrono::high_resolution_clock::now();
    // auto t3_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count();
    // std::cout << "Step 3 (Digit Recognition) took: " << t3_ms << " ms" << std::endl;

    // Fill grid with detected numbers
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (puzzleGrid[i][j] > 0) {
                sudokuGrid[i][j].number = puzzleGrid[i][j];
                sudokuGrid[i][j].mask = 1;
            }
        }
    }

    // --- 4. Sudoku Solver ---
    SudokuSolver solver;
    std::vector<std::vector<int>> solvedGrid;
    // auto t4_start = std::chrono::high_resolution_clock::now();
    bool success = solver.solve(puzzleGrid, solvedGrid);
    // auto t4_end = std::chrono::high_resolution_clock::now();
    // auto t4_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4_end - t4_start).count();
    // std::cout << "Step 4 (Sudoku Solving) took: " << t4_ms << " ms" << std::endl;

    // If solved, update numbers in the single grid; preserve mask so original detections are still marked.
    if (success) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (sudokuGrid[i][j].number == 0) {
                    sudokuGrid[i][j].number = solvedGrid[i][j];
                }
            }
        }
    } else {
        return SudokuResult{ sudokuGrid, SudokuStatus::NotSolved };
    }

    return SudokuResult{ sudokuGrid, SudokuStatus::Solved };
}

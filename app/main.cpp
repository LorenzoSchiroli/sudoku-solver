#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // added for timing

#include "sudoku_detection.hpp"
#include "grid_extraction.hpp"
#include "digit_recognition.hpp"
#include "sudoku_solver.hpp"

void printSudokuGrid(const std::vector<std::vector<int>>& grid) {
    std::cout << "-------------------------" << std::endl;
    for (size_t i = 0; i < grid.size(); ++i) {
        if (i % 3 == 0 && i != 0) std::cout << "-------------------------" << std::endl;
        for (size_t j = 0; j < grid[i].size(); ++j) {
            if (j % 3 == 0 && j != 0) std::cout << "| ";
            std::cout << grid[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <path_to_image>" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    // Load Image
    cv::Mat originalImage = cv::imread(imagePath);
    if (originalImage.empty()) {
        std::cerr << "Error: Could not read image at " << imagePath << std::endl;
        return -1;
    }

    // --- 1. Sudoku Detection ---
    auto t1_start = std::chrono::high_resolution_clock::now();
    std::optional<cv::Mat> boardCrop = detectSudokuBoard(originalImage);
    auto t1_end = std::chrono::high_resolution_clock::now();
    auto t1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count();
    std::cout << "Step 1 (Sudoku Detection) took: " << t1_ms << " ms" << std::endl;

    // Check Detection
    if (!boardCrop.has_value()) {
        std::cout << "board not detected" << std::endl;
        return 0;
    }

    // --- 2. Grid Extraction ---
    auto t2_start = std::chrono::high_resolution_clock::now();
    // Returns 9x9 grid of cell images (empty cells are handled internally by your function)
    std::vector<std::vector<cv::Mat>> cellCrops = analyze_sudoku_board(*boardCrop);
    auto t2_end = std::chrono::high_resolution_clock::now();
    auto t2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count();
    std::cout << "Step 2 (Grid Extraction) took: " << t2_ms << " ms" << std::endl;

    // --- 3. Digit Recognition ---
    auto t3_start = std::chrono::high_resolution_clock::now();
    DigitRecognizer recognizer(modelPath);
    std::vector<std::vector<int>> puzzleGrid = recognizer.predictGrid(cellCrops);
    auto t3_end = std::chrono::high_resolution_clock::now();
    auto t3_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count();
    std::cout << "Step 3 (Digit Recognition) took: " << t3_ms << " ms" << std::endl;

    std::cout << "Detected Puzzle:" << std::endl;
    printSudokuGrid(puzzleGrid);

    // --- 4. Sudoku Solver ---
    SudokuSolver solver;
    std::vector<std::vector<int>> solvedGrid;
    auto t4_start = std::chrono::high_resolution_clock::now();
    bool success = solver.solve(puzzleGrid, solvedGrid);
    auto t4_end = std::chrono::high_resolution_clock::now();
    auto t4_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4_end - t4_start).count();
    std::cout << "Step 4 (Sudoku Solving) took: " << t4_ms << " ms" << std::endl;

    // Print Result
    if (success) {
        std::cout << "\nSolved Sudoku:" << std::endl;
        printSudokuGrid(solvedGrid);
    } else {
        std::cout << "\nCould not solve the sudoku (invalid configuration)." << std::endl;
    }

    return 0;
}
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono> // added for timing
#include <algorithm>
#include "sudoku_main.hpp"

void printSudokuGrid(const std::vector<std::vector<Cell>>& grid, bool hide = false) {
    std::cout << "-------------------------" << std::endl;
    for (size_t i = 0; i < grid.size(); ++i) {
        if (i % 3 == 0 && i != 0) std::cout << "-------------------------" << std::endl;
        for (size_t j = 0; j < grid[i].size(); ++j) {
            if (j % 3 == 0 && j != 0) std::cout << "| ";
            const Cell& cell = grid[i][j];
            int out = hide ? ((cell.mask == 0) ? 0 : cell.number) : cell.number;
            std::cout << out << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];

    // Load Image
    cv::Mat originalImage = cv::imread(imagePath);
    if (originalImage.empty()) {
        std::cerr << "Error: Could not read image at " << imagePath << std::endl;
        return -1;
    }

    SudokuMain processor; // loads recognition model
    SudokuResult sudokuGrid = processor.sudoku_img2grid(originalImage);

    if (sudokuGrid.status == SudokuStatus::GridNotFound) {
        std::cout << "grid not found" << std::endl;
        return 0;
    }

    auto& grid = sudokuGrid.grid;
    std::cout << "Detected grid:" << std::endl;
    printSudokuGrid(grid, true);

    if (sudokuGrid.status == SudokuStatus::NotSolved) {
        std::cout << "not solved, an error occured" << std::endl;
        return 0;
    }

    std::cout << "Solved grid:" << std::endl;
    printSudokuGrid(grid, false);

    return 0;
}
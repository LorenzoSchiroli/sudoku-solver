#include "sudoku_detection.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Adjusted usage: allow optional output prefix
    if (argc < 2 || argc > 3) {
        std::cout << "Usage: ./test_sudoku_detection <image_path> [out_prefix]" << std::endl;
        return -1;
    }

    std::string outPrefix = "sudoku_";
    if (argc == 3) outPrefix = argv[2];

    cv::Mat src = cv::imread(argv[1]);
    if (src.empty()) {
        std::cout << "Failed to load image." << std::endl;
        return 0;
    }

    auto boards = detectSudokuBoards(src);
    auto saved = saveBoards(boards, outPrefix);
    
    if (saved.empty()) {
        std::cout << "No Sudoku boards found." << std::endl;
        return 0;
    }

    for (const auto& f : saved) std::cout << "Saved: " << f << std::endl;

    return 0;
}

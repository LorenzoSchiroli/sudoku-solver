#include "sudoku_detection.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector> // added

// New: save a series of Mats and return the filenames written
std::vector<std::string> saveBoards(const std::vector<cv::Mat>& boards, const std::string& outPrefix) {
    std::vector<std::string> filenames;
    int count = 0;
    for (const auto& b : boards) {
        std::string filename = outPrefix + std::to_string(++count) + ".png";
        cv::imwrite(filename, b);
        filenames.push_back(filename);
    }
    return filenames;
}

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

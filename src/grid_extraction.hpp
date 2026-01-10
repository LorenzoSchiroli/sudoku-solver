#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

std::vector<std::vector<cv::Mat>> analyze_sudoku_board(const cv::Mat& image);

// Utility to print the grid matrix (declared so test programs can use it)
void print_matrix(const std::vector<std::vector<cv::Mat>>& matrix);

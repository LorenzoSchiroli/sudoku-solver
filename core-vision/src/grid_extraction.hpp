#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

std::vector<std::vector<cv::Mat>> analyze_sudoku_board(const cv::Mat& image);

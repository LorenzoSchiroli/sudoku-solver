#pragma once

#include <optional>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

std::vector<cv::Mat> detectSudokuBoards(const cv::Mat& src);
std::optional<cv::Mat> detectSudokuBoard(const cv::Mat& src);
std::vector<std::string> saveBoards(const std::vector<cv::Mat>& boards, const std::string& outPrefix = "sudoku_");

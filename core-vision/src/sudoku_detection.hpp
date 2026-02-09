#pragma once

#include <optional>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

/**
 * Detect all Sudoku boards in the provided image and return each board as a
 * perspective-corrected `cv::Mat`.
 *
 * @param src Input BGR image.
 * @return Vector of warped board images (may be empty).
 */
std::vector<cv::Mat> detectSudokuBoards(const cv::Mat& src);

/**
 * Convenience helper that returns the first detected board or `std::nullopt`.
 */
std::optional<cv::Mat> detectSudokuBoard(const cv::Mat& src);

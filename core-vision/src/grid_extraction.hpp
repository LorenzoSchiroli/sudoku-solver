#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

/**
 * Analyze a full Sudoku board image and extract a 9x9 grid of cell crops.
 *
 * Each returned cell is a color `cv::Mat` containing the detected digit region.
 * Empty cells are represented as an empty `cv::Mat`.
 *
 * @param image Input color image containing a single Sudoku board.
 * @return 9x9 matrix of cell images (rows x cols). Returns empty vector on error.
 */
std::vector<std::vector<cv::Mat>> analyze_sudoku_board(const cv::Mat& image);

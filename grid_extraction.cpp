#include "grid_extraction.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <algorithm> // New header for safe min/max

// Define the size of the Sudoku grid
const int GRID_SIZE = 9;

/**
 * Determines if a cell contains a number by checking the density of edge pixels 
 * in its central region against a defined minimum ratio.
 * * @param cell_roi The region of interest (already edge-detected).
 * @param min_ratio The minimum ratio of edge pixels to total pixels for the cell to be 'full'.
 * @return True if the pixel count exceeds the ratio threshold (-1), False otherwise (0).
 */
bool is_cell_full(const cv::Mat& cell_roi, double min_ratio = 0.005) {
    // 1. Define a Central Region (Half the cell size)
    // Margin is 1/4 of the dimension on all sides (leaving 1/2 the size in the center).
    // This is mathematically equivalent to checking half the size.
    int margin = std::min(cell_roi.rows, cell_roi.cols) / 3;
    
    // Calculate the central region for analysis
    cv::Rect center_region(margin, margin, 
                           cell_roi.cols - 2 * margin, 
                           cell_roi.rows - 2 * margin);

    // No fallback: Trust the calculation based on your requirement
    cv::Mat check_area = cell_roi(center_region);
    
    // 2. Count the Non-Zero (Edge) Pixels
    int edge_pixel_count = cv::countNonZero(check_area); 

    // 3. Calculate Ratio and Check Threshold
    double total_pixels = check_area.total();
    double current_ratio = edge_pixel_count / total_pixels;

    if (current_ratio > min_ratio) {
        std::string filename = "cell.png";
        cv::imwrite(filename, check_area);
    }

    // Use an empirically determined ratio (0.005 or 0.5%) as a starting point.
    // This will flag a cell as 'full' if more than 0.5% of its central pixels are edges.
    return current_ratio > min_ratio; 
}

void removeBorders(cv::Mat& binaryImg){
    CV_Assert(binaryImg.type() == CV_8UC1);

    // Mask must be 2 pixels larger than the image
    cv::Mat mask(binaryImg.rows + 2, binaryImg.cols + 2, CV_8U, cv::Scalar(0));

    const int rows = binaryImg.rows;
    const int cols = binaryImg.cols;

    // Top & bottom rows
    for (int x = 0; x < cols; ++x) {
        if (binaryImg.at<uchar>(0, x) == 255)
            cv::floodFill(binaryImg, mask, {x, 0}, 0);
        if (binaryImg.at<uchar>(rows - 1, x) == 255)
            cv::floodFill(binaryImg, mask, {x, rows - 1}, 0);
    }

    // Left & right columns
    for (int y = 0; y < rows; ++y) {
        if (binaryImg.at<uchar>(y, 0) == 255)
            cv::floodFill(binaryImg, mask, {0, y}, 0);
        if (binaryImg.at<uchar>(y, cols - 1) == 255)
            cv::floodFill(binaryImg, mask, {cols - 1, y}, 0);
    }
}

cv::Mat cropToContainingContour(cv::Mat& binaryImg) {
    removeBorders(binaryImg);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImg.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 1. Define the specific point to check (Center of the rect)
    cv::Point2f targetPoint(
        (binaryImg.cols - 1) / 2.0f, // Width is image.cols
        (binaryImg.rows - 1) / 2.0f  // Height is image.rows
    );

    for (const auto& contour : contours) {
        // 3. Stage 1: Fast Bounding Box Check
        // If the point isn't even in the square, don't bother calculating polygon geometry
        cv::Rect r = cv::boundingRect(contour);
        
        if (r.contains(targetPoint)) {
            cv::Rect cropRegion = r & cv::Rect(0, 0, binaryImg.cols, binaryImg.rows);
            return binaryImg(cropRegion).clone();
        }
    }

    // Return empty if no contour contains the point
    return cv::Mat();
}

// Replace old cropToContainingContour with new signature/implementation:
cv::Rect cropToContainingContour(const cv::Mat& binaryImg, const cv::Rect& region) {
    // Validate and clamp region
    cv::Rect bounded = region & cv::Rect(0, 0, binaryImg.cols, binaryImg.rows);
    if (bounded.empty()) return cv::Rect();

    // Operate on a copy of the binary subregion (so the original binaryImg isn't modified)
    cv::Mat cell = binaryImg(bounded).clone();
    cv::imwrite("cell.png", cell);
    removeBorders(cell);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cell.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return cv::Rect(); // empty -> no contours found
    }

    // Check center point of the cell-subregion
    cv::Point2f targetPoint((cell.cols - 1) / 2.0f, (cell.rows - 1) / 2.0f);

    // Find the contour whose bounding box is closest to the center point
    int bestIdx = -1;
    double bestDist = std::numeric_limits<double>::max();
    for (int i = 0; i < contours.size(); ++i)
    {
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 == 0) continue;
        cv::Point2f c(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        double d = cv::norm(c - targetPoint);
        if (d < bestDist)
        {
            bestDist = d;
            bestIdx = i;
        }
    }

    // Empty contours
    if (bestIdx == -1) {
        return cv::Rect(); // empty -> no contours found
    }

    const auto& contour = contours[bestIdx];
    cv::Rect r = cv::boundingRect(contour);
    // Shift local rect back to full-image coordinates
    cv::Rect cropRegion(r.x + bounded.x, r.y + bounded.y, r.width, r.height);
    // Keep within global image bounds
    cropRegion &= cv::Rect(0, 0, binaryImg.cols, binaryImg.rows);
    return cropRegion;
    // for (const auto& contour : contours) {
    //     cv::Rect r = cv::boundingRect(contour);
    //     if (r.contains(targetPoint)) {
            
    //     }
    // }

    // return cv::Rect(); // empty -> no containing contour found
}

// New: pad, make square, and clamp the region to image bounds
cv::Rect addPadding(const cv::Rect& region, const cv::Size& imgSize, int pad = 3) {
	if (region.empty()) return cv::Rect();

	// Start with padded bbox
	int paddedW = region.width + 2 * pad;
	int paddedH = region.height + 2 * pad;

	// Side length to make square
	int side = std::max(paddedW, paddedH);

	// Ensure side does not exceed image dimensions
	side = std::min(side, std::min(imgSize.width, imgSize.height));

	// Center the square on the original region center
	int centerX = region.x + region.width / 2;
	int centerY = region.y + region.height / 2;

	int newX = centerX - side / 2;
	int newY = centerY - side / 2;

	// Clamp to image bounds
	newX = std::max(0, std::min(newX, imgSize.width - side));
	newY = std::max(0, std::min(newY, imgSize.height - side));

	return cv::Rect(newX, newY, side, side);
}

// Main function for Sudoku analysis
std::vector<std::vector<cv::Mat>> analyze_sudoku_board(const cv::Mat& input_img) {
    // 1. Validate and convert to grayscale (grayscale conversion remains inside this function)
    if (input_img.empty() || input_img.channels() != 3) {
        std::cerr << "ERROR: Empty or wrong image passed to analyze_sudoku_board()" << std::endl;
        return {};
    }

    cv::Mat img_color = input_img.clone();
    cv::resize(img_color, img_color, cv::Size(512, 512));

    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(img_gray, img_gray, cv::Size(9, 9), 0);
    // Perform Canny Edge Detection on the entire image
    cv::Mat img_edges;
    cv::adaptiveThreshold(img_gray, img_edges, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 17, 4);
    // cv::threshold(img_gray, img_edges, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);


    std::string filename = "sudoku_edge.png";
    cv::imwrite(filename, img_edges);


    // Store cell crops; empty cv::Mat means empty cell
    std::vector<std::vector<cv::Mat>> sudoku_cells(GRID_SIZE, std::vector<cv::Mat>(GRID_SIZE));

    int img_width = img_edges.cols;
    int img_height = img_edges.rows;

    // --- Geometric Calculation based on Centering/Squareness ---
    // int min_dim = std::min(img_width, img_height);
    double cell_size_x = (double)img_width / GRID_SIZE;
    double cell_size_y = (double)img_height / GRID_SIZE;
    // double x_start = (img_width - min_dim) / 2.0;
    // double y_start = (img_height - min_dim) / 2.0;
    // ------------------------------------------------------------------

    // 2. Iterate through all 81 cells
    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            // Calculate ROI coordinates
            int x = static_cast<int>(col * cell_size_x);
            int y = static_cast<int>(row * cell_size_y);
            int w = static_cast<int>(cell_size_x);
            int h = static_cast<int>(cell_size_y);

            // enlarge
            int padding = 10;
            x = std::max(0, x - padding);
            y = std::max(0, y - padding);
            w = std::min(w + padding*2, img_width - x);
            h = std::min(h + padding*2, img_height - y);

            cv::Rect cell_rect(x, y, w, h);
            cv::Mat cell_roi_bw = img_edges(cell_rect);

            if (is_cell_full(cell_roi_bw, 0.1)) {
                // Get crop region (in full-image coords) from the binary image and cell coordinates
                cv::Rect cropRegion = cropToContainingContour(img_edges, cell_rect);
                if (cropRegion.area() > 0) {
                    // cropRegion = cell_rect;
                    // Add padding and make square (clamped to image)
                    cropRegion = addPadding(cropRegion, img_color.size(), 3);
                    // Extract the color crop from the original color image
                    cv::Mat cell_roi = img_color(cropRegion).clone();
                    sudoku_cells[row][col] = cell_roi;
                }
            } else {
                // leave as empty cv::Mat
            }
        }
    }

    std::filesystem::path cells_dir("cells");
    try {
        if (!std::filesystem::exists(cells_dir)) {
            std::filesystem::create_directory(cells_dir);
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(cells_dir)) {
                std::error_code ec;
                std::filesystem::remove_all(entry.path(), ec);
                if (ec) {
                    std::cerr << "WARNING: Failed to remove " << entry.path()
                                << ": " << ec.message() << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Filesystem operation failed: " << e.what() << std::endl;
    }
    for (size_t r = 0; r < sudoku_cells.size(); ++r) {
        for (size_t c = 0; c < sudoku_cells[r].size(); ++c) {
            const cv::Mat& mat = sudoku_cells[r][c];
            if (!mat.empty()) {
                std::string filename = "cell_" + std::to_string(r) + "_" + std::to_string(c) + ".png";
                std::filesystem::path outpath = cells_dir / filename;
                cv::imwrite(outpath.string(), mat);
            }
        }
    }

    return sudoku_cells;
}
 
// Utility function to print the resulting matrix (now checks if cv::Mat is empty)
void print_matrix(const std::vector<std::vector<cv::Mat>>& matrix) {
    std::cout << "\nSudoku Matrix (-1: Full, 0: Empty):\n";
    for (const auto& row : matrix) {
        for (const auto& mat : row) {
            std::cout << (mat.empty() ? " 0" : " -1") << " ";
        }
        std::cout << "\n";
    }
}
 
// main removed: moved to `test_grid_extraction.cpp` for library-style usage

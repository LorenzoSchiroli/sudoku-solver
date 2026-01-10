#include "grid_extraction.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image.png>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: Could not open or find the image: " << image_path << std::endl;
        return -1;
    }

    auto result_matrix = analyze_sudoku_board(img);

    if (!result_matrix.empty()) {
        print_matrix(result_matrix);

        // Ensure 'cells' directory exists and clear older images
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

        for (size_t r = 0; r < result_matrix.size(); ++r) {
            for (size_t c = 0; c < result_matrix[r].size(); ++c) {
                const cv::Mat& mat = result_matrix[r][c];
                if (!mat.empty()) {
                    std::string filename = "cell_" + std::to_string(r) + "_" + std::to_string(c) + ".png";
                    std::filesystem::path outpath = cells_dir / filename;
                    cv::imwrite(outpath.string(), mat);
                }
            }
        }
    }

    return 0;
}

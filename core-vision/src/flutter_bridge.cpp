#define SUDOKU_BRIDGE_IMPLEMENTATION
#include "flutter_bridge.hpp"

#include "sudoku_main.hpp"
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
#include <vector>
#include <memory>
// #include <fstream>

extern "C" {

void* sudoku_create_processor(const char* model_path) {
    try {
        if (model_path) {
            return new SudokuMain(std::string(model_path));
        } else {
            return new SudokuMain();
        }
    } catch (...) {
        return nullptr;
    }
}

void sudoku_destroy_processor(void* processor) {
    if (!processor) return;
    delete static_cast<SudokuMain*>(processor);
}

static SudokuResultC* make_empty_result(SudokuStatusC status) {
    SudokuResultC* r = new (std::nothrow) SudokuResultC();
    // if (!r) return nullptr;
    r->status = static_cast<int32_t>(status);
    for (int i = 0; i < 81; ++i) {
        r->cells[i].number = 0;
        r->cells[i].mask = 0;
    }
    return r;
}

static SudokuResultC* convert_result(const SudokuResult& cppRes) {
    SudokuResultC* out = new (std::nothrow) SudokuResultC();

    out->status = static_cast<int32_t>(cppRes.status == SudokuStatus::GridNotFound ? SudokuStatus_GridNotFound :
                                       cppRes.status == SudokuStatus::NotSolved ? SudokuStatus_NotSolved :
                                       SudokuStatus_Solved);
    
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            const Cell& cell = cppRes.grid[r][c];
            int idx = r * 9 + c;
            out->cells[idx].number = cell.number;
            out->cells[idx].mask = cell.mask;
        }
    }
    return out;
}

SudokuResultC* sudoku_process_image_bytes(void* processor, const uint8_t* data, size_t length) {
    if (!processor || !data || length == 0) return make_empty_result(SudokuStatus_InitializationError);
    try {
        SudokuMain* proc = static_cast<SudokuMain*>(processor);
        cv::Mat rawData(1, length, CV_8UC1, const_cast<uint8_t*>(data));
        cv::Mat img = cv::imdecode(rawData, cv::IMREAD_COLOR);
        if (img.empty()) return make_empty_result(SudokuStatus_GridNotFound);
        SudokuResult cppRes = proc->sudoku_img2grid(img);
        return convert_result(cppRes);
    } catch (...) {
        return make_empty_result(SudokuStatus_InitializationError);
    }
}

void sudoku_free_result(SudokuResultC* result) {
    delete result;
}

}

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

class DigitRecognizer {
public:
    DigitRecognizer(const std::string& modelPath);
    std::vector<std::pair<std::string, int>> predictFolder(const std::string& folderPath);

    // Changed: accept cv::Mat instead of a file path
    int predictSingle(const cv::Mat& image);

    // Changed: accept cv::Mat instead of a file path; grayscale conversion happens here
    bool preprocessImageToTensor(const cv::Mat& inputImage, std::vector<float>& inputTensorValues);
    int getArgMax(const std::vector<float>& array);

    // Add predictGrid: takes a 2D grid of cv::Mat and returns a 2D grid of predictions.
    std::vector<std::vector<int>> predictGrid(const std::vector<std::vector<cv::Mat>>& grid);

private:
    // --- MOVE THESE TO THE TOP ---
    const int inputWidth = 32;
    const int inputHeight = 32;
    const int inputChannels = 3;

    // Now these can safely use the variables above because they are declared afterwards
    std::vector<int64_t> inputShape;
    size_t inputTensorSize;
    
    // ... rest of your variables (env, session, etc.)
    Ort::Env env;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;
    std::string inputNameStr;
    std::string outputNameStr;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    Ort::MemoryInfo memoryInfo;
};
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

    std::vector<std::vector<int>> predictGrid(const std::vector<std::vector<cv::Mat>>& grid);
    std::vector<std::pair<std::string, int>> predictFolder(const std::string& folderPath);

private:
    // Keep other prediction helpers private (available for internal use)
    std::vector<int> predictList(const std::vector<cv::Mat>& images);
    std::vector<int> predictBatch(const std::vector<cv::Mat>& batchImages);

    // Fixed declaration (no class-scope qualifier) and kept private
    bool preprocessToBuffer(const cv::Mat& inputImage, float* dst);
    int getArgMax(const std::vector<float>& array);

    // --- MOVE THESE TO THE TOP ---
    const int inputWidth = 32;
    const int inputHeight = 32;
    const int inputChannels = 3;
    const int batchSize = 8;

    // Now these can safely use the variables above because they are declared afterwards
    std::vector<int64_t> inputShape;
    size_t inputTensorSize;
    size_t singleImageTensorSize;
    
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
#include "digit_recognition.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // Required for blobFromImageds

namespace fs = std::filesystem;

DigitRecognizer::DigitRecognizer(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "MnistInference"),
      memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    // Allocator needed for string management
    Ort::AllocatorWithDefaultOptions allocator;

    // Get Input Name
    auto inputNamePtr = session->GetInputNameAllocated(0, allocator);
    inputNameStr = inputNamePtr.get();
    inputNames = { inputNameStr.c_str() };

    // Get Output Name
    auto outputNamePtr = session->GetOutputNameAllocated(0, allocator);
    outputNameStr = outputNamePtr.get();
    outputNames = { outputNameStr.c_str() };

    // Set shapes: [Batch, Channel, Height, Width]
    inputShape = {1, inputChannels, inputHeight, inputWidth};
    inputTensorSize = inputChannels * inputHeight * inputWidth;
}

std::vector<std::pair<std::string, int>> DigitRecognizer::predictFolder(const std::string& folderPath) {
    std::vector<std::pair<std::string, int>> results;
    std::vector<fs::path> files;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) files.push_back(entry.path());
        }
    } catch (...) { return results; }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });

    for (const auto& path : files) {
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR); // Force load as BGR
        if (img.empty()) continue;

        int pred = predictSingle(img);
        if (pred >= 0) results.emplace_back(path.filename().string(), pred);
    }
    return results;
}

int DigitRecognizer::predictSingle(const cv::Mat& image) {
    if (image.empty()) return 0;

    std::vector<float> inputTensorValues;
    
    // 1. Preprocess using OpenCV DNN blob (Handles Resizing, float conversion, and HWC->CHW swap)
    if (!preprocessImageToTensor(image, inputTensorValues)) return -1;

    // 2. Create Tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputShape.data(), inputShape.size()
    );

    try {
        // 3. Run Inference
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1
        );

        // 4. Get Result
        float* floatArr = outputTensors.front().GetTensorMutableData<float>();
        size_t count = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        return getArgMax(std::vector<float>(floatArr, floatArr + count));

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    }
}

bool DigitRecognizer::preprocessImageToTensor(const cv::Mat& inputImage, std::vector<float>& inputTensorValues) {
    cv::Mat processed;

    // 1. Handle Channels: Convert to BGR if Gray or BGRA
    if (inputImage.channels() == 1) {
        cv::cvtColor(inputImage, processed, cv::COLOR_GRAY2BGR);
    } else if (inputImage.channels() == 4) {
        cv::cvtColor(inputImage, processed, cv::COLOR_BGRA2BGR);
    } else {
        processed = inputImage.clone();
    }

    // 2. Resize and Pad (Letterbox) to keep aspect ratio
    int canvasH = inputHeight;
    int canvasW = inputWidth;
    
    float scale = std::min((float)canvasW / processed.cols, (float)canvasH / processed.rows);
    int newW = std::max(1, (int)(processed.cols * scale));
    int newH = std::max(1, (int)(processed.rows * scale));

    cv::resize(processed, processed, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);

    // Create black canvas
    cv::Mat padded = cv::Mat::zeros(canvasH, canvasW, CV_8UC3);
    
    // Center image on canvas
    int x_offset = (canvasW - newW) / 2;
    int y_offset = (canvasH - newH) / 2;
    processed.copyTo(padded(cv::Rect(x_offset, y_offset, newW, newH)));

    // 3. Convert to Blob (HWC -> CHW, BGR->RGB, Scale 0-1)
    // swapRB = true (converts BGR to RGB), crop = false
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0/255.0, cv::Size(canvasW, canvasH), cv::Scalar(), true, false);

    // 4. Flatten to vector
    if (blob.isContinuous()) {
        inputTensorValues.assign((float*)blob.data, (float*)blob.data + blob.total());
    } else {
        return false;
    }

    return true;
}

int DigitRecognizer::getArgMax(const std::vector<float>& array) {
    if (array.empty()) return 0;
    // pick prediction except 0
    auto maxIt = std::max_element(array.begin() + 1, array.end());
    return (int)std::distance(array.begin(), maxIt);
}

std::vector<std::vector<int>> DigitRecognizer::predictGrid(const std::vector<std::vector<cv::Mat>>& grid) {
    std::vector<std::vector<int>> results;
    results.reserve(grid.size());
    for (const auto& row : grid) {
        std::vector<int> rowResults;
        rowResults.reserve(row.size());
        for (const auto& cell : row) {
            rowResults.push_back(predictSingle(cell)); // Returns -1 on error, which is safe enough or map to 0
        }
        results.push_back(std::move(rowResults));
    }
    return results;
}
#include "digit_recognition.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 

namespace fs = std::filesystem;

DigitRecognizer::DigitRecognizer(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "MnistInference"),
      memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNamePtr = session->GetInputNameAllocated(0, allocator);
    inputNameStr = inputNamePtr.get();
    inputNames = { inputNameStr.c_str() };

    auto outputNamePtr = session->GetOutputNameAllocated(0, allocator);
    outputNameStr = outputNamePtr.get();
    outputNames = { outputNameStr.c_str() };

    // Shapes
    inputShape = { batchSize, inputChannels, inputHeight, inputWidth };
    inputTensorSize = batchSize * inputChannels * inputHeight * inputWidth;
    singleImageTensorSize = inputChannels * inputHeight * inputWidth;
}

std::vector<int> DigitRecognizer::predictList(const std::vector<cv::Mat>& images) {
    // 1. Initialize results with default 0
    std::vector<int> allPredictions(images.size(), 0);

    // Buffers to hold current batch and their original indices
    std::vector<cv::Mat> batchBuffer;
    std::vector<size_t> originalIndices;
    batchBuffer.reserve(batchSize);
    originalIndices.reserve(batchSize);

    for (size_t i = 0; i < images.size(); ++i) {
        // If empty, we skip adding to batch. 
        // The result is already 0 in allPredictions[i] by default.
        if (images[i].empty()) {
            continue; 
        }

        batchBuffer.push_back(images[i]);
        originalIndices.push_back(i);

        // If batch is full, execute
        if (batchBuffer.size() == batchSize) {
            std::vector<int> batchResults = predictBatch(batchBuffer);
            
            // Map results back to original positions
            for (size_t k = 0; k < batchResults.size(); ++k) {
                allPredictions[originalIndices[k]] = batchResults[k];
            }
            
            batchBuffer.clear();
            originalIndices.clear();
        }
    }

    // Process remaining items (partial batch)
    if (!batchBuffer.empty()) {
        std::vector<int> batchResults = predictBatch(batchBuffer);
        for (size_t k = 0; k < batchResults.size(); ++k) {
            allPredictions[originalIndices[k]] = batchResults[k];
        }
    }

    return allPredictions;
}

std::vector<int> DigitRecognizer::predictBatch(const std::vector<cv::Mat>& batchImages) {
    // 1. Prepare fixed-size input tensor
    // We allocate the FULL fixed batch size (filled with 0.0f)
    std::vector<float> inputTensorValues(inputTensorSize, 0.0f); 

    // 2. Fill the buffer
    // Loop through the fixed batch size (0 to batchSize-1)
    for (int i = 0; i < batchSize; ++i) {
        // Calculate offset for this image slot
        float* dstPtr = inputTensorValues.data() + (i * singleImageTensorSize);

        // If we have an image for this slot, process it.
        // If i >= batchImages.size(), we just leave zeros (padding).
        if (i < batchImages.size()) {
             // We assume batchImages only contains valid mats now (filtered by predictList),
             // but checking !empty() is good safety.
            if (!batchImages[i].empty()) {
                preprocessToBuffer(batchImages[i], dstPtr);
            }
        }
    }

    // 3. Create Tensor
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputShape.data(), inputShape.size()
    );

    std::vector<int> results;
    results.reserve(batchImages.size());

    try {
        // 4. Run Inference
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1
        );

        // 5. Parse Output
        float* floatArr = outputTensors.front().GetTensorMutableData<float>();
        size_t totalElements = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        size_t numClasses = totalElements / batchSize;

        // Retrieve results ONLY for the actual input images
        for (size_t i = 0; i < batchImages.size(); ++i) {
            float* imgResultStart = floatArr + (i * numClasses);
            
            // Safety check
            if (batchImages[i].empty()) {
                results.push_back(0); 
            } else {
                results.push_back(getArgMax(std::vector<float>(imgResultStart, imgResultStart + numClasses)));
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        results.assign(batchImages.size(), 0);
    }

    return results;
}

bool DigitRecognizer::preprocessToBuffer(const cv::Mat& inputImage, float* dst) {
    cv::Mat processed;

    if (inputImage.channels() == 1) {
        cv::cvtColor(inputImage, processed, cv::COLOR_GRAY2BGR);
    } else if (inputImage.channels() == 4) {
        cv::cvtColor(inputImage, processed, cv::COLOR_BGRA2BGR);
    } else {
        processed = inputImage.clone();
    }

    int canvasH = inputHeight;
    int canvasW = inputWidth;
    
    float scale = std::min((float)canvasW / processed.cols, (float)canvasH / processed.rows);
    int newW = std::max(1, (int)(processed.cols * scale));
    int newH = std::max(1, (int)(processed.rows * scale));

    cv::resize(processed, processed, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);

    cv::Mat padded = cv::Mat::zeros(canvasH, canvasW, CV_8UC3);
    int x_offset = (canvasW - newW) / 2;
    int y_offset = (canvasH - newH) / 2;
    processed.copyTo(padded(cv::Rect(x_offset, y_offset, newW, newH)));

    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0/255.0, cv::Size(canvasW, canvasH), cv::Scalar(), true, false);

    if (blob.isContinuous()) {
        std::memcpy(dst, blob.data, blob.total() * sizeof(float));
        return true;
    } 
    return false;
}

std::vector<std::pair<std::string, int>> DigitRecognizer::predictFolder(const std::string& folderPath) {
    std::vector<fs::path> files;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) files.push_back(entry.path());
        }
    } catch (...) { return {}; }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });

    std::vector<cv::Mat> images;
    std::vector<std::string> filenames;
    images.reserve(files.size());
    filenames.reserve(files.size());

    for (const auto& path : files) {
        // Load all images, even if empty/failed
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
        images.push_back(img); 
        filenames.push_back(path.filename().string());
    }

    // predictList handles empty images by returning 0
    std::vector<int> predictions = predictList(images);

    std::vector<std::pair<std::string, int>> results;
    results.reserve(predictions.size());
    for(size_t i = 0; i < predictions.size(); ++i) {
        // We include all results, even 0s (which might mean empty image)
        results.emplace_back(filenames[i], predictions[i]);
    }
    return results;
}

std::vector<std::vector<int>> DigitRecognizer::predictGrid(const std::vector<std::vector<cv::Mat>>& grid) {
    // 1. Flatten
    std::vector<cv::Mat> flatImages;
    for (const auto& row : grid) {
        flatImages.insert(flatImages.end(), row.begin(), row.end());
    }

    // 2. Predict (Handles empty mats automatically)
    std::vector<int> flatResults = predictList(flatImages);

    // 3. Reconstruct
    std::vector<std::vector<int>> gridResults;
    gridResults.reserve(grid.size());
    
    size_t idx = 0;
    for (const auto& row : grid) {
        std::vector<int> rowResults;
        rowResults.reserve(row.size());
        for (size_t c = 0; c < row.size(); ++c) {
            rowResults.push_back(flatResults[idx++]);
        }
        gridResults.push_back(std::move(rowResults));
    }
    return gridResults;
}

int DigitRecognizer::getArgMax(const std::vector<float>& array) {
    if (array.empty()) return 0;
    auto maxIt = std::max_element(array.begin() + 1, array.end());
    return (int)std::distance(array.begin(), maxIt);
}
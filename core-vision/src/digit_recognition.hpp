#pragma once

#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

/**
 * DigitRecognizer
 *
 * Lightweight wrapper around an ONNX digit-recognition model. Provides
 * convenience methods to predict digits for a single 9x9 grid, a folder
 * of images, or a flat list/batch of images. Internal helpers manage
 * preprocessing and batching for ONNX Runtime.
 */
class DigitRecognizer {
public:
  /**
   * Load the ONNX model from `modelPath` and prepare the runtime session.
   */
  DigitRecognizer(const std::string &modelPath);

  /**
   * Predict an entire 9x9 grid of cell images. Empty cv::Mat entries
   * are handled and will yield a prediction of 0.
   */
  std::vector<std::vector<int>>
  predictGrid(const std::vector<std::vector<cv::Mat>> &grid);

  /**
   * Predict all image files in a folder. Returns pairs of (filename, label).
   * Images that fail to load are returned with label 0.
   */
  std::vector<std::pair<std::string, int>>
  predictFolder(const std::string &folderPath);

private:
  // Internal helpers: operate on lists and fixed-size batches
  std::vector<int> predictList(const std::vector<cv::Mat> &images);
  std::vector<int> predictBatch(const std::vector<cv::Mat> &batchImages);

  // Preprocess a single image into the provided float buffer (row-major, CHW)
  bool preprocessToBuffer(const cv::Mat &inputImage, float *dst);
  // Return index of maximum element in the array (0 when empty)
  int getArgMax(const std::vector<float> &array);

  // ONNX input tensor geometry and batching
  const int inputWidth = 32;
  const int inputHeight = 32;
  const int inputChannels = 3;
  const int batchSize = 8;

  std::vector<int64_t> inputShape;
  size_t inputTensorSize;
  size_t singleImageTensorSize;

  // ONNX Runtime state
  Ort::Env env;
  Ort::AllocatorWithDefaultOptions allocator;
  std::unique_ptr<Ort::Session> session;
  std::string inputNameStr;
  std::string outputNameStr;
  std::vector<const char *> inputNames;
  std::vector<const char *> outputNames;
  Ort::MemoryInfo memoryInfo;
};
#include "digit_recognition.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <path_to_images_folder>" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string folderPath = argv[2];

    try {
        DigitRecognizer recognizer(modelPath);
        auto results = recognizer.predictFolder(folderPath);

        for (const auto& kv : results) {
            const auto& filename = kv.first;
            int pred = kv.second;
            std::cout << filename << ": " << pred << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

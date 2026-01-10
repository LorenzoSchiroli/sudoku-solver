#include "sudoku_detection.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <optional>

using namespace cv;
using namespace std;

// Helper to order points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
vector<Point2f> orderPoints(const vector<Point>& pts) {
    vector<Point2f> sortedPts(4);
    vector<Point2f> origPts;
    for (const auto& p : pts) origPts.push_back(Point2f((float)p.x, (float)p.y));

    // Sort by Y to separate top and bottom
    sort(origPts.begin(), origPts.end(), [](Point2f a, Point2f b) { return a.y < b.y; });

    // Top points (smallest Y)
    vector<Point2f> topPts = {origPts[0], origPts[1]};
    sort(topPts.begin(), topPts.end(), [](Point2f a, Point2f b) { return a.x < b.x; });
    sortedPts[0] = topPts[0]; // TL
    sortedPts[1] = topPts[1]; // TR

    // Bottom points (largest Y)
    vector<Point2f> bottomPts = {origPts[2], origPts[3]};
    sort(bottomPts.begin(), bottomPts.end(), [](Point2f a, Point2f b) { return a.x < b.x; });
    sortedPts[3] = bottomPts[0]; // BL
    sortedPts[2] = bottomPts[1]; // BR

    return sortedPts;
}

// New: detect sudoku boards in a given image and return each warped board as a Mat
vector<Mat> detectSudokuBoards(const Mat& src) {
    vector<Mat> boards;
    if (src.empty()) return boards;

    Mat gray, blurred, thresh;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(9, 9), 0);

    // Adaptive threshold is crucial for lighting variations
    adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

    // Find contours
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Sort contours by area (descending)
    sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
        return contourArea(a) > contourArea(b);
    });

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < 1000) continue; // Filter small noise

        double peri = arcLength(cnt, true);
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.02 * peri, true);

        if (approx.size() == 4 && isContourConvex(approx)) {
            vector<Point2f> ordered = orderPoints(approx);

            float wA = norm(ordered[2] - ordered[3]);
            float wB = norm(ordered[1] - ordered[0]);
            int maxWidth = static_cast<int>(max(wA, wB));

            float hA = norm(ordered[1] - ordered[2]);
            float hB = norm(ordered[0] - ordered[3]);
            int maxHeight = static_cast<int>(max(hA, hB));

            vector<Point2f> dstPts = {
                {0, 0},
                {static_cast<float>(maxWidth - 1), 0},
                {static_cast<float>(maxWidth - 1), static_cast<float>(maxHeight - 1)},
                {0, static_cast<float>(maxHeight - 1)}
            };

            Mat M = getPerspectiveTransform(ordered, dstPts);
            Mat output;
            warpPerspective(src, output, M, Size(maxWidth, maxHeight));

            boards.push_back(output);
        }
    }
    return boards;
}

// New: helper that returns the first detected board or std::nullopt if none
std::optional<Mat> detectSudokuBoard(const Mat& src) {
    auto boards = detectSudokuBoards(src);
    if (boards.empty()) return std::nullopt;
    return boards.front();
}

// New: save a series of Mats and return the filenames written
vector<string> saveBoards(const vector<Mat>& boards, const string& outPrefix) {
    vector<string> filenames;
    int count = 0;
    for (const auto& b : boards) {
        string filename = outPrefix + to_string(++count) + ".png";
        imwrite(filename, b);
        filenames.push_back(filename);
    }
    return filenames;
}

// main removed: moved to `test_sudoku_detection.cpp` for library-style usage

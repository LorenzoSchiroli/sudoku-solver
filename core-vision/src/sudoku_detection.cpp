#include "sudoku_detection.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

/**
 * Order 4 points into top-left, top-right, bottom-right, bottom-left.
 */
vector<Point2f> orderPoints(const vector<Point> &pts) {
  vector<Point2f> sortedPts(4);
  vector<Point2f> origPts;
  for (const auto &p : pts)
    origPts.push_back(Point2f((float)p.x, (float)p.y));

  // Sort by Y to separate top and bottom
  sort(origPts.begin(), origPts.end(),
       [](Point2f a, Point2f b) { return a.y < b.y; });

  // Top points (smallest Y)
  vector<Point2f> topPts = {origPts[0], origPts[1]};
  sort(topPts.begin(), topPts.end(),
       [](Point2f a, Point2f b) { return a.x < b.x; });
  sortedPts[0] = topPts[0]; // TL
  sortedPts[1] = topPts[1]; // TR

  // Bottom points (largest Y)
  vector<Point2f> bottomPts = {origPts[2], origPts[3]};
  sort(bottomPts.begin(), bottomPts.end(),
       [](Point2f a, Point2f b) { return a.x < b.x; });
  sortedPts[3] = bottomPts[0]; // BL
  sortedPts[2] = bottomPts[1]; // BR

  return sortedPts;
}

/**
 * Detect quadrilateral contours that resemble Sudoku boards, warp and return
 * them.
 */
vector<Mat> detectSudokuBoards(const Mat &src) {
  vector<Mat> boards;
  if (src.empty())
    return boards;

  Mat gray, blurred, thresh, resized;
  float targetSize = 1000.0f;
  float scale = targetSize / std::max(src.cols, src.rows);
  resize(src, resized, Size(), scale, scale, INTER_AREA);
  cvtColor(resized, gray, COLOR_BGR2GRAY);
  GaussianBlur(gray, blurred, Size(9, 9), 0);

  // Adaptive threshold is crucial for lighting variations
  adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY_INV, 29, 3);

  // Save threshold image for debugging/inspection
  // try {
  //     imwrite("thresh.png", thresh);
  // } catch (const cv::Exception& e) {
  //     cerr << "Warning: failed to write threshold image: " << e.what() <<
  //     endl;
  // }

  // Find contours
  vector<vector<Point>> contours;
  findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  // double minArea = 1000.0; // Set your threshold

  // contours.erase(std::remove_if(contours.begin(), contours.end(),
  // [minArea](const std::vector<cv::Point>& c) {
  //     return cv::contourArea(c) < minArea;
  // }), contours.end());

  // // Save a simple debug image showing all detected contours
  // try {
  //     Mat contourVis;
  //     cvtColor(thresh, contourVis, COLOR_GRAY2BGR);
  //     drawContours(contourVis, contours, -1, Scalar(0, 0, 255), 2, LINE_AA);
  //     imwrite("detected_contours.png", contourVis);
  // } catch (const cv::Exception& e) {
  //     cerr << "Warning: failed to write contour debug image: " << e.what() <<
  //     endl;
  // }

  // Sort contours by area (descending)
  sort(contours.begin(), contours.end(),
       [](const vector<Point> &a, const vector<Point> &b) {
         return contourArea(a) > contourArea(b);
       });

  for (const auto &cnt : contours) {
    double area = contourArea(cnt);
    if (area < 1000)
      continue; // Filter small noise

    double peri = arcLength(cnt, true);
    vector<Point> approx;
    approxPolyDP(cnt, approx, 0.05 * peri, true);

    if (approx.size() == 4 && isContourConvex(approx)) {

      bool touchesEdge = false;
      int margin = 2; // px margin
      for (const auto &p : approx) {
        if (p.x <= margin || p.x >= resized.cols - margin || p.y <= margin ||
            p.y >= resized.rows - margin) {
          touchesEdge = true;
          break;
        }
      }
      if (touchesEdge)
        continue; // Skip boards that aren't fully contained

      vector<Point2f> orderedLowRes = orderPoints(approx);

      vector<Point2f> ordered;
      for (auto &p : orderedLowRes) {
        ordered.push_back(Point2f(p.x / scale, p.y / scale));
      }

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
          {0, static_cast<float>(maxHeight - 1)}};

      Mat M = getPerspectiveTransform(ordered, dstPts);
      Mat output;
      warpPerspective(src, output, M, Size(maxWidth, maxHeight));

      boards.push_back(output);
    }
  }

  // // Save the first detected board for debugging/inspection
  // if (!boards.empty()) {
  //     try {
  //         imwrite("first_board.png", boards.front());
  //     } catch (const cv::Exception& e) {
  //         cerr << "Warning: failed to write first board image: " << e.what()
  //         << endl;
  //     }
  // }

  return boards;
}

/**
 * Convenience helper returning the first detected board or std::nullopt.
 */
std::optional<Mat> detectSudokuBoard(const Mat &src) {
  auto boards = detectSudokuBoards(src);
  if (boards.empty())
    return std::nullopt;
  return boards.front();
}

#ifndef CAMERATRACKER_HPP
#define CAMERATRACKER_HPP

#include <vector>

#include "obzerver/circular_buffer.hpp"
#include "opencv2/core/core.hpp"

typedef std::vector<cv::Point2f>  pts_vec_t;

class CameraTracker
{
protected:
  bool initialized;
  std::size_t hist_len;
  CircularBuffer<cv::Mat> camera_hist;

  cv::Mat cache_diff_image;
  cv::Mat cache_sof_image;

  CircularBuffer<cv::Mat> frame_hist; // Min size: 2
  CircularBuffer<pts_vec_t> pts_hist; // Min size: 2

public:
  CameraTracker(const std::size_t hist_len);
  bool Update(const cv::Mat& frame_rgb);
};

#endif // CAMERATRACKER_HPP

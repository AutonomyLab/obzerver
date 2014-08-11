#ifndef CAMERATRACKER_HPP
#define CAMERATRACKER_HPP

#include <vector>

#include "obzerver/circular_buffer.hpp"
#include "obzerver/benchmarker.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

typedef std::vector<cv::Point2f>  pts_vec_t;

struct KeypointsGreaterThan {
    bool operator() (const cv::KeyPoint& k1, const cv::KeyPoint& k2) {
        return k1.response > k2.response;
    }
};

class CameraTracker
{
protected:
  bool initialized;
  std::size_t hist_len;
  CircularBuffer<cv::Mat> camera_hist;
  cv::Ptr<cv::FeatureDetector> feature_detector;

  cv::Mat cache_frame_stablized;
  cv::Mat cache_frame_diff;
  cv::Mat cache_sof_image;

  CircularBuffer<cv::Mat> frame_gray_hist; // Min size: 2

  std::vector<cv::KeyPoint> kpts;
  pts_vec_t tracked_features_prev;
  pts_vec_t tracked_features_curr;
  pts_vec_t tracked_features_curr_stab;
  pts_vec_t detected_features_prev;
  pts_vec_t detected_features_curr;
  std::vector<uchar> tracking_status;

  cv::Mat est_transform;
  cv::Mat est_outliers;

  std::size_t max_features;
  std::size_t pylk_winsize;
  unsigned int pylk_iters;
  double pylk_eps;

  StepBenchmarker& ticker;
  void UpdateDiff();
  void UpdateSOF();
public:
  CameraTracker(const std::size_t hist_len,
                const cv::Ptr<cv::FeatureDetector> feature_detector,
                const std::size_t max_features,
                const std::size_t pylk_winsize,
                unsigned int pylk_iters = 30,
                double pylk_eps = 0.01);
  bool Update(const cv::Mat& frame_gray);

  const cv::Mat& LatestDiff() const;
  const cv::Mat& LalestSOF() const;
  const pts_vec_t& LatestFeatures() const;
};

#endif // CAMERATRACKER_HPP

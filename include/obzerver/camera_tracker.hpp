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
  /*
    camera_transform_hist: (begin -> end)
    t->t-1 t-1->t-2 ... t-(len-2)->t-(len-1)

    camera_transform_hist_acc:
    index i: transform from t-i -> t-(len-1)
  */
  CircularBuffer<cv::Mat> camera_transform_hist;
  std::vector<cv::Mat> camera_transform_hist_acc;
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

  cv::Mat est_homography_transform;
  cv::Mat est_homography_outliers;

  std::size_t max_features;
  std::size_t pylk_winsize;
  unsigned int pylk_iters;
  double pylk_eps;

  StepBenchmarker& ticker;
  void UpdateDiff();
  void UpdateSOF();
  void UpdateAccumulatedTransforms();
  void FailureUpdate();
public:
  CameraTracker(const std::size_t hist_len,
                const cv::Ptr<cv::FeatureDetector> feature_detector,
                const std::size_t max_features,
                const std::size_t pylk_winsize,
                unsigned int pylk_iters = 30,
                double pylk_eps = 0.01);
  bool Update(const cv::Mat& frame_gray);

  const cv::Mat& GetLatestDiff() const {return cache_frame_diff;}
  const cv::Mat& GetLatestSOF() const {return cache_sof_image;}
  const cv::Mat& GetStablized() const {return cache_frame_stablized;}
  const pts_vec_t& GetTrackedFeaturesCurr() const {return tracked_features_curr;}
  const pts_vec_t& GetTrackedFeaturesPrev() const {return tracked_features_prev;}
  const cv::Mat& GetHomographyOutliers() const {return est_homography_outliers;}
  const cv::Mat& GetLatestCameraTransform() const {return camera_transform_hist.latest(); }
};

#endif // CAMERATRACKER_HPP

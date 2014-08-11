#include <algorithm>

#include "glog/logging.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "obzerver/camera_tracker.hpp"

CameraTracker::CameraTracker(const std::size_t hist_len,
                             const cv::Ptr<cv::FeatureDetector> feature_detector,
                             const std::size_t max_features,
                             const std::size_t pylk_winsize,
                             unsigned int pylk_iters,
                             double pylk_eps):
  initialized(false),
  hist_len(hist_len),
  camera_hist(hist_len),
  feature_detector(feature_detector),
  frame_gray_hist(2),
  max_features(max_features),
  pylk_winsize(pylk_winsize),
  pylk_iters(pylk_iters),
  pylk_eps(pylk_eps),
  ticker(StepBenchmarker::GetInstance())

{
  ;
}

bool CameraTracker::Update(const cv::Mat &frame_gray) {
  frame_gray_hist.push_front(frame_gray);
  ticker.tick("  [CT] Frame Copy");
  if (!initialized) {
    camera_hist.push_front(cv::Mat::eye(3, 3, CV_64FC1)); // TODO: Check the type
    initialized = true;
    return false;
  }

  kpts.clear();
  feature_detector->detect(frame_gray_hist.prev(), kpts);
  ticker.tick("  [CT] Feature Detection");
  LOG(INFO) << "[CT] Keypoints: " << kpts.size();
  std::size_t pts_to_copy = 0;

  if (kpts.size() > max_features) {
    std::sort(kpts.begin(), kpts.end(), KeypointsGreaterThan());
    pts_to_copy = max_features;
  } else {
    pts_to_copy = kpts.size();
  }

  ticker.tick("  [CT] Sorting");
  LOG(INFO) << "[CT] Keypoints to copy: " << pts_to_copy;

  detected_features_prev.resize(pts_to_copy);
  for (size_t id = 0; id < pts_to_copy; id++) {
    detected_features_prev[id] = kpts.at(id).pt;
  }

  tracking_status.clear();

  if (detected_features_prev.size()) {
    cv::calcOpticalFlowPyrLK(
          frame_gray_hist.prev(),
          frame_gray_hist.latest(),
          detected_features_prev,
          detected_features_curr,
          tracking_status,
          cv::noArray(),
          cv::Size(pylk_winsize, pylk_winsize),
          3,
          cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, pylk_iters, pylk_eps),
          cv::OPTFLOW_LK_GET_MIN_EIGENVALS);
    tracked_features_prev.clear();
    tracked_features_curr.clear();
    for (size_t j = 0; j < tracking_status.size(); j++) {
      if (tracking_status[j]) {
        tracked_features_prev.push_back(detected_features_prev[j]);
        tracked_features_curr.push_back(detected_features_curr[j]);
      }
    }
  } else {
    camera_hist.push_front(cv::Mat::eye(3, 3, CV_64FC1));
    LOG(WARNING) << "[CT] No feature points to track";
    return false;
  }
  ticker.tick("  [CT] Feature Tracking");
  LOG(INFO) << "[CT] Tracked Features: " << tracked_features_curr.size();
  if (tracked_features_curr.size() > 10) {
    est_transform = cv::findHomography(
          tracked_features_curr,
          tracked_features_prev,
          CV_LMEDS,
          1.0,
          est_outliers
          );

    cv::warpPerspective(frame_gray_hist.latest(),
                        cache_frame_stablized,
                        est_transform,
                        frame_gray_hist.latest().size(),
                        cv::INTER_CUBIC
                        );

    camera_hist.push_front(est_transform);
    ticker.tick("  [CT] Find Homography");
  } else {
    camera_hist.push_front(cv::Mat::eye(3, 3, CV_64FC1));
    LOG(WARNING) << "[CT] Not enough feature points to do stablization";
    return false;
  }

  UpdateDiff();
  UpdateSOF();
  return true;
}

void CameraTracker::UpdateDiff() {
  cv::Scalar thres_mean;
  cv::Scalar thres_stddev;

  cv::absdiff(cache_frame_stablized, frame_gray_hist.prev(), cache_frame_diff);
  cv::meanStdDev(cache_frame_diff, thres_mean, thres_stddev);
  cv::threshold(cache_frame_diff, cache_frame_diff, thres_mean[0] + thres_stddev[0], 0, cv::THRESH_TOZERO);
}

// TODO: Move all params
void CameraTracker::UpdateSOF() {
  tracked_features_curr_stab.clear();
  cache_sof_image = cv::Mat::zeros(cache_frame_stablized.rows, cache_frame_stablized.cols, CV_32FC1);
  cv::perspectiveTransform(tracked_features_curr, tracked_features_curr_stab, camera_hist.latest());
  for (unsigned int i = 0; i < tracked_features_curr_stab.size(); i++) {
    if (!est_outliers.data || 1 == est_outliers.at<uchar>(i, 0)) continue; // Skip the inliers
    const cv::Point2f d = tracked_features_curr_stab[i] - tracked_features_prev[i];
    if (tracked_features_curr_stab[i].x >= 0 && tracked_features_curr_stab[i].y >=0 && tracked_features_curr_stab[i].x < cache_sof_image.cols && tracked_features_curr_stab[i].y < cache_sof_image.rows) {
      // Let's assume that there is a max and min displacment for each point
      double mag  = sqrt(d.x*d.x + d.y*d.y);
      if (mag > 0.1 && mag < 10.0) {
        cv::circle(cache_sof_image, tracked_features_curr_stab[i], mag*5, CV_RGB(mag*10, mag*10, mag*10), -1);
      }
    }
  }
}

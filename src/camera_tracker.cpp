#include <algorithm>

#include "glog/logging.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include "obzerver/utility.hpp"
#include "obzerver/camera_tracker.hpp"

namespace obz
{

CameraTracker::CameraTracker(const std::size_t hist_len,
                             const cv::Ptr<cv::FeatureDetector> feature_detector,
                             const std::size_t max_features,
                             const std::size_t pylk_winsize,
                             unsigned int pylk_iters,
                             double pylk_eps):
  initialized(false),
  hist_len(hist_len),
  camera_transform_hist(hist_len),
  feature_detector(feature_detector),
  frame_gray_hist(2),
  diff_hist(3),
  max_features(max_features),
  pylk_winsize(pylk_winsize),
  pylk_iters(pylk_iters),
  pylk_eps(pylk_eps),
  ticker(StepBenchmarker::GetInstance())

{
  ;
}

bool CameraTracker::Update(const cv::Mat &frame_gray, const cv::Mat &frame_rgb) {
  frame_gray_hist.push_front(frame_gray.clone());
  ticker.tick("CT_Frame_Copy");
  if (!initialized) {
    initialized = true;
    FailureUpdate();
    return false;
  }

  kpts.clear();
  feature_detector->detect(frame_gray_hist.prev(), kpts);
  ticker.tick("CT_Feature_Detection");
  LOG(INFO) << "[CT] Keypoints: " << kpts.size();
  std::size_t pts_to_copy = 0;

  if (kpts.size() > max_features) {
    std::sort(kpts.begin(), kpts.end(), KeypointsGreaterThan());
    pts_to_copy = max_features;
  } else {
    pts_to_copy = kpts.size();
  }

  ticker.tick("CT_Sorting");
  LOG(INFO) << "[CT] Keypoints to copy: " << pts_to_copy;

  detected_features_prev.resize(pts_to_copy);
  for (size_t id = 0; id < pts_to_copy; id++) {
    detected_features_prev[id] = kpts.at(id).pt;
  }

  tracking_status.clear();
  if (detected_features_prev.size()) {
    detected_features_curr.clear();
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
    LOG(WARNING) << "[CT] No feature points to track";
    FailureUpdate();
    return false;
  }
  ticker.tick("CT_Feature_Tracking");
  LOG(INFO) << "[CT] Tracked Features: " << tracked_features_curr.size();
  if (tracked_features_curr.size() > 10) {
    est_homography_transform = cv::findHomography(
          tracked_features_curr,
          tracked_features_prev,
          CV_LMEDS,
          1.0,
          est_homography_outliers
          );

    cv::warpPerspective(frame_gray_hist.latest(),
                        cache_frame_stablized_gray,
                        est_homography_transform,
                        frame_gray_hist.latest().size(),
                        cv::INTER_CUBIC,
                        cv::BORDER_TRANSPARENT
                        );

    if (frame_rgb.cols > 0 && frame_rgb.rows > 0) {
      cv::warpPerspective(frame_rgb,
                          cache_frame_stablized_rgb,
                          est_homography_transform,
                          frame_gray_hist.latest().size(),
                          cv::INTER_CUBIC,
                          cv::BORDER_TRANSPARENT
                          );
    }

    camera_transform_hist.push_front(est_homography_transform.clone());
    ticker.tick("CT_Find_Homography");
  } else {
    LOG(WARNING) << "[CT] Not enough feature points to do stablization";
    FailureUpdate();
    return false;
  }

  UpdateDiff();
  //UpdateSOF();
  ticker.tick("CT_Update_Diff_&_SOF");

  UpdateAccumulatedTransforms();
  ticker.tick("CT_Acc_Camera_Trans");
  return true;
}

void CameraTracker::UpdateDiff() {
  cv::Scalar thres_mean;
  cv::Scalar thres_stddev;

  cv::Mat diff;
  cv::absdiff(cache_frame_stablized_gray, frame_gray_hist.prev(), diff);
  //cv::absdiff(frame_gray_hist.latest(), frame_gray_hist.prev(), diff);
  cv::meanStdDev(diff, thres_mean, thres_stddev);
  cv::threshold(diff, diff, thres_mean[0] + thres_stddev[0], 255, cv::THRESH_TOZERO);
  //cv::threshold(diff, diff, 1, 255, cv::THRESH_BINARY);
  //cv::threshold(diff, diff, 0, 0, cv::THRESH_TOZERO | cv::THRESH_OTSU);

  cache_frame_diff = diff; return;

  diff_hist.push_front(diff);
  cache_frame_diff = cv::Mat::zeros(diff.size(), CV_16UC1);
  //cv::Mat trans_to_histlen = camera_transform_hist_acc.at(0).inv();
  //cv::Mat dummy;
  for (unsigned int i = 0; i < diff_hist.size(); i++) {
    //cv::addWeighted(cache_frame_diff, 0.2, diff, 0.8, 0, cache_frame_diff);
    //cv::Mat trans_to_current = camera_transform_hist_acc.at(i) * trans_to_histlen;
    //cv::warpPerspective(diff_hist.at(i), dummy, trans_to_current, diff_hist.at(i).size());
    //cv::add(cache_frame_diff, dummy, cache_frame_diff, cv::noArray(), 1);
    cv::add(cache_frame_diff, diff_hist.at(i), cache_frame_diff, cv::noArray(), 1);
    //cv::bitwise_and(cache_frame_diff, dummy, cache_frame_diff);
  }

  cache_frame_diff = cache_frame_diff * (1.0 / diff_hist.size());
}

// TODO: Move all params
void CameraTracker::UpdateSOF() {
  tracked_features_curr_stab.clear();
  cache_sof_image = cv::Mat::zeros(cache_frame_stablized_gray.rows, cache_frame_stablized_gray.cols, CV_32FC1);
  cv::perspectiveTransform(tracked_features_curr, tracked_features_curr_stab, camera_transform_hist.latest());
  for (unsigned int i = 0; i < tracked_features_curr_stab.size(); i++) {
    if (!est_homography_outliers.data || 1 == est_homography_outliers.at<uchar>(i, 0)) continue; // Skip the inliers
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

// CTHA[i] = From Frame t-i to frame t-histlen+1
void CameraTracker::UpdateAccumulatedTransforms() {
  camera_transform_hist_acc.resize(camera_transform_hist.size());
  // Skip the last element, it should eye(3,3)
  unsigned int index = camera_transform_hist_acc.size() - 1;  
  camera_transform_hist_acc[index] = cv::Mat::eye(3, 3, CV_64FC1);
  for (CircularBuffer<cv::Mat>::const_reverse_iterator it = camera_transform_hist.rbegin() + 1 ;
       it != camera_transform_hist.rend(); it++) {
      index--;
      camera_transform_hist_acc[index] = (*it) * camera_transform_hist_acc[index + 1];
  }
}

void CameraTracker::FailureUpdate() {
  camera_transform_hist.push_front(cv::Mat::eye(3, 3, CV_64FC1)); // TODO: Check the type
  cache_frame_diff = cv::Mat::zeros(frame_gray_hist.latest().size(), CV_8UC1);
  cache_sof_image = cv::Mat::zeros(frame_gray_hist.latest().size(), CV_8UC1);
  cache_frame_stablized_gray = frame_gray_hist.latest().clone();
  UpdateAccumulatedTransforms();
}

}  // namespace obz

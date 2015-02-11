#include "obzerver/kalman.hpp"
#include "obzerver/utility.hpp"

#include "opencv2/core/core.hpp"
#include "glog/logging.h"

#include <vector>

namespace obz
{

KalmanFilter::KalmanFilter(const cv::Rect &r):
  kf_(6, 4, 0),
  state_(6, 1, CV_32F),
  measurement_(4, 1, CV_32F),
  last_update_time_(0)
{

  kf_.transitionMatrix = *(cv::Mat_<float>(6,6) <<
                           1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

  kf_.measurementMatrix = *(cv::Mat_<float>(4, 6) <<
                            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  Init(r);
}

KalmanFilter::~KalmanFilter()
{}

void KalmanFilter::Init(const cv::Rect &r)
{
  last_update_time_ = cv::getTickCount();
  measurement_.setTo(cv::Scalar(0.0));

  state_(0, 0) = r.x;
  state_(1, 0) = r.y;
  state_(2, 0) = 0.0;
  state_(3, 0) = 0.0;
  state_(4, 0) = r.width;
  state_(5, 0) = r.height;

  kf_.statePre = state_.clone();
  kf_.statePost = state_.clone();

  setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-4));
  setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
  setIdentity(kf_.errorCovPre, cv::Scalar::all(0.1));
  setIdentity(kf_.errorCovPost, cv::Scalar::all(0.1));
}

void KalmanFilter::Predict(const cv::Mat& camera_transform)
{
  const double dt =
      static_cast<double>(cv::getTickCount() - last_update_time_) / cv::getTickFrequency();

  last_update_time_ = cv::getTickCount();

  kf_.transitionMatrix.at<float>(0, 2) = dt;
  kf_.transitionMatrix.at<float>(1, 3) = dt;

  // Compensate Camera Motion
  // TODO: Merge this into Kalman Filter Model
  cv::Point2f pt_stab(kf_.statePost.at<float>(0, 0), kf_.statePost.at<float>(1, 0));
  pt_stab = obz::util::TransformPoint(pt_stab, camera_transform);
  kf_.statePost.at<float>(0, 0) = pt_stab.x;
  kf_.statePost.at<float>(1, 0) = pt_stab.y;

  state_ = kf_.predict();
}

obz::object_t KalmanFilter::Update(const cv::Mat &camera_transform)
{
  Predict(camera_transform);
  return obz::object_t(state_);
}

obz::object_t KalmanFilter::Update(const cv::Mat &camera_transform, const cv::Rect &obz)
{
  Predict(camera_transform);
  measurement_(0, 0) = obz.x;
  measurement_(1, 0) = obz.y;
  measurement_(2, 0) = obz.width;
  measurement_(3, 0) = obz.height;
  state_ = kf_.correct(measurement_);
  return obz::object_t(state_);
}

}  // namespace obz

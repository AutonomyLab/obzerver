#ifndef KALMAN_HPP
#define KALMAN_HPP

#include <obzerver/common_types.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <cstdint>

namespace obz
{

class KalmanFilter
{
public:  

private:
  cv::KalmanFilter kf_;

  cv::Mat_<float> state_;  // x, y, vx, vy, w, h

  cv::Mat_<float> measurement_; // x, y, w, h

  std::int64_t last_update_time_;

  void Predict(const cv::Mat& camera_transform);
public:
  explicit KalmanFilter(const cv::Rect& r = cv::Rect(0, 0, 0, 0));
  ~KalmanFilter();

  // Also performs the reset
  void Init(const cv::Rect& r);

  // Update without measurment (just prediction)
  obz::object_t Update(const cv::Mat& camera_transform);

  obz::object_t Update(const cv::Mat& camera_transform, const cv::Rect& obz);

  const cv::Mat_<float>& GetState() const {return state_;}
};

}  // namespace obz
#endif

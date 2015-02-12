#ifndef MULTI_OBJECT_TRACKER_HPP
#define MULTI_OBJECT_TRACKER_HPP

#include "obzerver/common_types.hpp"
#include "obzerver/kalman.hpp"
#include "obzerver/tobject.hpp"
#include "obzerver/hungarian.hpp"

#include <opencv2/core/core.hpp>
#include <cstdint>
#include <list>

namespace obz
{

struct Track
{
  std::uint32_t uid;
  cv::Ptr<obz::TObject> object_ptr;
  cv::Ptr<obz::ExKalmanFilter> ekf_ptr;
  std::uint32_t skipped_frames;

  Track(const std::uint32_t uid = 0):
    uid(uid), skipped_frames(0) {}

  const cv::Rect& GetBB() const {return object_ptr->Get().bb;}
};

class MultiObjectTracker
{
private:
  std::uint32_t next_uid_;
  // TODO: Optimize this
  std::vector<Track> tracks_;
  std::uint32_t history_len_; // for TObject
  float fps_;  // for TObject
  const std::uint32_t max_skipped_frames_;   // To lose track

  obz::alg::Hungarian hungarian_solver_;

  void CreateTrack(const cv::Rect& bb, const cv::Mat& frame);
  void DeleteTrack(const std::uint32_t track_index);

  // W/O obzervation
  void UpdateTrack(const std::uint32_t track_index, const cv::Mat& frame, const cv::Mat& camera_transform);

  // W/ obzervation
  void UpdateTrack(const std::uint32_t track_index, const cv::Rect& bb, const cv::Mat& frame, const cv::Mat& camera_transform);

public:
  MultiObjectTracker(const std::uint32_t hist_len,
                     const float fps,
                     const std::uint32_t max_skipped_frames);
  ~MultiObjectTracker();

  // camera_transform is 3x3 homogenous transform to go from
  // frame t-1's coordinate system to frame t
  void Update(const obz::rect_vec_t& detections,
              const cv::Mat& frame,
              const cv::Mat& camera_transform);

  const std::vector<obz::Track>& GetTracks() const {return tracks_;}

  void DrawTracks(cv::Mat& frame);
};
}

#endif

#ifndef MULTI_OBJECT_TRACKER_HPP
#define MULTI_OBJECT_TRACKER_HPP

#include "obzerver/common_types.hpp"
#include "obzerver/kalman.hpp"
#include "obzerver/tobject.hpp"
#include "obzerver/hungarian.hpp"

#include <opencv2/core/core.hpp>
#include <cstdint>
#include <list>
#include <map>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace obz
{

struct Track
{
  std::uint32_t uid;
  cv::Ptr<obz::TObject> object_ptr;
  cv::Ptr<obz::ExKalmanFilter> ekf_ptr;
  cv::Mat self_similarity_rendered;
  float dom_freq;
  std::vector<float> avg_spectrum;
  std::uint32_t skipped_frames;

//  enum periodicity_state_t
//  {
//    PS_UNKNOWN = 0,
//    PS_NON_PERIODIC,
//    PS_TESTING,
//    PS_PERIODIC,
//    PS_NUM
//  } periodicity_state;

  Track(const std::uint32_t uid = 0):
    uid(uid), skipped_frames(0) {}

  Track(const Track& rhs):
    uid(rhs.uid),
    object_ptr(rhs.object_ptr),
    ekf_ptr(rhs.ekf_ptr),
    self_similarity_rendered(rhs.self_similarity_rendered),
    dom_freq(rhs.dom_freq),
    avg_spectrum(rhs.avg_spectrum),
    skipped_frames(rhs.skipped_frames) {}

  const cv::Rect& GetBB() const {return object_ptr->Get().bb;}
};

// Forward Dec
class MultiObjectTracker;

class PeriodicityWorkerThread
{
private:
  obz::MultiObjectTracker& mot;
public:
  PeriodicityWorkerThread(obz::MultiObjectTracker& mot);

  void operator ()();
};

class MultiObjectTracker
{
private:

  // uid = 0 is reserved for unknown
  std::uint32_t next_uid_;
  // TODO: Optimize this
  std::vector<Track> tracks_;
  std::uint32_t history_len_; // for TObject
  float fps_;  // for TObject
  const std::uint32_t max_skipped_frames_;   // To lose track

  friend class PeriodicityWorkerThread;
  // -1 not set
  std::atomic<std::int32_t> focus_track_index_;
  std::condition_variable condition_;
  std::atomic<bool> clear_track_;
  std::atomic<bool> terminate_;
  std::mutex mutex_;
  std::thread per_thread_;
  bool IsFree() const {return focus_track_index_ == -1; }
  bool SetTrack(std::uint32_t track_index);
  bool ClearTrack();
  void Terminate();


  void CreateTrack(const cv::Rect& bb, const cv::Mat& frame);
  void DeleteTrack(const std::uint32_t track_index);

  // W/O obzervation
  void UpdateTrack(const std::uint32_t track_index,
                   const cv::Mat& frame,
                   const cv::Mat& camera_transform,
                   const bool calc_self_similarity);

  // W/ obzervation
  void UpdateTrack(const std::uint32_t track_index,
                   const cv::Rect& bb,
                   const cv::Mat& frame,
                   const cv::Mat& camera_transform,
                   const bool calc_self_similarity);

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

  std::size_t GetNumTracks() const {return tracks_.size();}
  const std::vector<obz::Track>& GetTracks() const {return tracks_;}

  void DrawTracks(cv::Mat& frame);
};
}

#endif

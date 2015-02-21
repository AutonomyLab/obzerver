#ifndef MULTI_OBJECT_TRACKER_HPP
#define MULTI_OBJECT_TRACKER_HPP

#include "obzerver/common_types.hpp"
#include "obzerver/kalman.hpp"
#include "obzerver/tobject.hpp"
#include "obzerver/hungarian.hpp"

#include <opencv2/core/core.hpp>
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
  static const std::uint32_t UNKNOWN;
  // 0 is reserved
  std::uint32_t uid;
  cv::Ptr<obz::TObject> object_ptr;
  cv::Ptr<obz::ExKalmanFilter> ekf_ptr;
  cv::Ptr<obz::SelfSimilarity> ss_ptr;
  cv::Ptr<obz::Periodicity> per_ptr;
  cv::Mat self_similarity_rendered;
  float dom_freq;
  std::vector<float> avg_spectrum;
  std::uint32_t skipped_frames;
  std::uint32_t life;

//  enum periodicity_state_t
//  {
//    PS_UNKNOWN = 0,
//    PS_NON_PERIODIC,
//    PS_TESTING,
//    PS_PERIODIC,
//    PS_NUM
//  } periodicity_state;

  Track(const std::uint32_t uid = UNKNOWN):
    uid(uid), dom_freq(-1.0), skipped_frames(0), life(0) {}

  const cv::Rect& GetBB() const {return object_ptr->Get().bb;}
};

// uid -> Track
typedef std::pair<uint32_t, Track> track_pair_t;
typedef std::map<uint32_t, Track> tracks_map_t;

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
  std::atomic<std::uint64_t> current_time;
  // uid = 0 is reserved
  std::uint32_t next_uid_;
  // uid->track
  tracks_map_t tracks_;
  // [0..tracks_.size]->uid
  std::vector<std::uint32_t> tracks_indexes_;

  std::uint32_t history_len_; // for TObject
  float fps_;  // for TObject
  const std::uint32_t max_skipped_frames_;   // To lose track
  std::uint64_t focus_order_;

  /* Periodicity Thread Stuff */
  friend class PeriodicityWorkerThread;
  std::atomic<std::uint32_t> focus_track_uid_;  // -1 not set
  std::condition_variable pwt_condition_;
  std::atomic<bool> pwt_terminate_;
  std::mutex pwt_mutex_;
  std::thread pwt_thread_;
  bool IsPWTFree() const {return false && focus_track_uid_ == Track::UNKNOWN; }
  bool SetPWTTrack(std::uint32_t track_uid);
  bool ClearPWTTrack();
  void TerminatePWT();

  std::uint32_t FindTrackIndex(const std::uint32_t track_uid);
  void CreateTrack(const cv::Rect& bb, const cv::Mat& frame, const cv::Mat& diff_frame, const float flow);
  bool DeleteTrackIfStale(const std::uint32_t track_uid);

  // W/O obzervation
  void UpdateTrack(const std::uint32_t track_uid,
                   const cv::Mat& frame,
                   const cv::Mat& diff_frame,
                   const cv::Mat& camera_transform);

  // W/ obzervation
  void UpdateTrack(const std::uint32_t track_uid,
                   const cv::Rect& bb,
                   const cv::Mat& frame,
                   const cv::Mat& diff_frame,
                   const float flow,
                   const cv::Mat& camera_transform);

public:
  MultiObjectTracker(const std::uint32_t hist_len,
                     const float fps,
                     const std::uint32_t max_skipped_frames);
  ~MultiObjectTracker();

  // camera_transform is 3x3 homogenous transform to go from
  // frame t-1's coordinate system to frame t
  void Update(const obz::rect_vec_t& detections, const std::vector<float> &flows,
              const cv::Mat& frame, const cv::Mat &diff_frame,
              const cv::Mat& camera_transform);

  std::size_t GetNumTracks() const {return tracks_.size();}
  const obz::tracks_map_t& GetTracks() const {return tracks_;}

  void DrawTracks(cv::Mat& frame);
};
}

#endif

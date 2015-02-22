#ifndef MULTI_OBJECT_TRACKER_HPP
#define MULTI_OBJECT_TRACKER_HPP

#include "obzerver/common_types.hpp"
#include "obzerver/kalman.hpp"
#include "obzerver/tobject.hpp"
#include "obzerver/hungarian.hpp"

#include <opencv2/core/core.hpp>

#include <map>
// C++11
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace obz
{

// TODO: Convert it into a class + Make a simple track_t datastructure for public usage
struct Track
{
  static const std::uint32_t UNKNOWN;
  // 0 is reserved
  std::uint32_t uid;
  std::uint32_t skipped_frames;
  std::uint32_t life;

  cv::Ptr<obz::TObject> object_ptr;
  cv::Ptr<obz::ExKalmanFilter> ekf_ptr;

  // This is used by SelfSimilarity thread to update the SS
  cv::Ptr<obz::SelfSimilarity> ss_ptr;
  cv::Mat self_similarity_rendered;

  // This is used by MOT to update Motion Periodicity
  cv::Ptr<obz::Periodicity> per_ptr;

  float dom_freq;
  std::vector<float> avg_spectrum;

  Track(const std::uint32_t uid = Track::UNKNOWN):
    uid(uid), skipped_frames(0), life(0), dom_freq(-1.0) {}

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
  obz::periodicity_method_t periodicity_method_;
  std::atomic<std::uint64_t> current_time_;
  // uid = 0 is reserved
  std::uint32_t next_uid_;
  // uid->track
  tracks_map_t tracks_;
  // [0..tracks_.size]->uid
  std::vector<std::uint32_t> tracks_indexes_;

  // for TObject
  std::uint32_t history_len_;
  float fps_;

  std::uint32_t max_skipped_frames_;
  float max_matching_cost_;

  // To permutate focus over all tracks
  std::uint64_t focus_order_;

  /* Periodicity Thread Stuff */
  friend class PeriodicityWorkerThread;
  std::atomic<std::uint32_t> focus_track_uid_;
  std::condition_variable pwt_condition_;
  std::atomic<bool> pwt_terminate_;
  std::mutex pwt_mutex_;
  std::thread pwt_thread_;
  bool IsPWTFree() const {return focus_track_uid_ == Track::UNKNOWN; }
  bool SetPWTTrack(std::uint32_t track_uid);
  void TerminatePWT();

  std::uint32_t FindTrackIndex(const std::uint32_t track_uid);
  void CreateTrack(const cv::Rect& bb, const cv::Mat& frame, const cv::Mat& diff_frame, const float flow);
  bool DeleteTrackIfStale(const std::uint32_t track_uid);

  // If bb.area() > 0 : Update KF with obzervation (bb, flow)
  // otherwise update KF without obzervation, just prediction
  void UpdateTrack(const std::uint32_t track_uid,
                   const cv::Mat& frame,
                   const cv::Mat& diff_frame,
                   const cv::Mat& camera_transform,
                   const cv::Rect& bb = cv::Rect(0, 0, 0, 0),
                   const float flow = 0.0);

public:
  MultiObjectTracker(
      const obz::periodicity_method_t periodicity_method,
      const std::uint32_t hist_len,
      const float fps,
      const std::uint32_t max_skipped_frames,
      const float max_matching_cost);
  ~MultiObjectTracker();

  // camera_transform is 3x3 homogenous transform to go from
  // frame t-1's coordinate system to frame t
  void Update(const obz::rect_vec_t& detections,
              const std::vector<float> &flows,
              const cv::Mat& frame,
              const cv::Mat &diff_frame,
              const cv::Mat& camera_transform);

  std::size_t GetNumTracks() const {return tracks_.size();}

  // This is not thread-safe when using SS
  const obz::tracks_map_t& GetTracks() const {return tracks_;}

  // This is thread-safe (clears and resizes the vector)
  std::size_t CopyAllTracks(std::vector<obz::Track>& tracks_vec);

  void DrawTracks(cv::Mat& frame, const bool verbose = false);
};
}

#endif

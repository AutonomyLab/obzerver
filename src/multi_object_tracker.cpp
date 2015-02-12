#include "obzerver/multi_object_tracker.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/utility.hpp"

#include "glog/logging.h"

#include <sstream>
namespace obz
{

MultiObjectTracker::MultiObjectTracker(const std::uint32_t history_len,
                                       const float fps,
                                       const std::uint32_t max_skipped_frames)
  : next_uid_(0),
    history_len_(history_len),
    fps_(fps),
    max_skipped_frames_(max_skipped_frames)
{
  hungarian_solver_.diag(false);
}

MultiObjectTracker::~MultiObjectTracker()
{}

void MultiObjectTracker::CreateTrack(const cv::Rect &bb, const cv::Mat &frame)
{
  LOG(INFO) << "Creating track " << next_uid_ << " for " << bb;

  // Create a track and append that to tracks_ vector
  tracks_.push_back(Track(next_uid_));

  // Modify the object
  Track& track = tracks_.back();

  track.object_ptr = cv::Ptr<obz::TObject>(new obz::TObject(history_len_, fps_));

  // TODO
  track.object_ptr->Update(bb, frame, true);
  track.ekf_ptr = cv::Ptr<obz::ExKalmanFilter>(new obz::ExKalmanFilter(bb));
  next_uid_++;
}

void MultiObjectTracker::DeleteTrack(const std::uint32_t track_index)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Deleting " << track_index << "'s track: " << tracks_[track_index].uid;

  // Check dstr calls
  tracks_.erase(tracks_.begin() + track_index);
}

void MultiObjectTracker::UpdateTrack(const std::uint32_t track_index,
                                     const cv::Mat &frame,
                                     const cv::Mat &camera_transform)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Updating w/o " << track_index << "'s track: " << tracks_[track_index].uid;

  tracks_[track_index].skipped_frames++;

  // Update kalman filter w/o prediction, pass the result to TObject
  obz::object_t obj = tracks_[track_index].ekf_ptr->Update(camera_transform);
  obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);

  tracks_[track_index].object_ptr->Update(obj, frame, true);
}

void MultiObjectTracker::UpdateTrack(
    const std::uint32_t track_index,
    const cv::Rect &bb,
    const cv::Mat &frame,
    const cv::Mat &camera_transform)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Updating obz " << track_index << "'s track: " << tracks_[track_index].uid;
  LOG(INFO) << "  with " << bb;

  tracks_[track_index].skipped_frames = 0;

  // Update kalman filter with bb, pass the result to TObject
  obz::object_t obj = tracks_[track_index].ekf_ptr->Update(bb, camera_transform);
  obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);
  tracks_[track_index].object_ptr->Update(obj, frame, true);
}

void MultiObjectTracker::Update(const rect_vec_t &detections,
                                const cv::Mat &frame,
                                const cv::Mat& camera_transform)
{
  const std::size_t n_dets = detections.size();
  const std::size_t n_tracks = tracks_.size();
  LOG(INFO) << "[MOT] Number of detections: " << n_dets << " Tracks: " << n_tracks;

  if (n_dets == 0)
  {
    LOG(INFO) << "[MOT] No detections";
    for (std::size_t t = 0; t < n_tracks; t++)
    {
      UpdateTrack(t, frame, camera_transform);
    }
    return;
  }

  if (n_tracks == 0)
  {
    LOG(INFO) << "[MOT] No tracks";
    // We should create tracks for all detections
    for (std::size_t d = 0; d < n_dets; d++)
    {
      CreateTrack(detections[d], frame);
    }
    return;
  }

  cv::Mat_<int> cost(n_tracks, n_dets);

  // Go over all tracks and detections to calculate assignment cost
  for (std::size_t t = 0; t < n_tracks; t++)
  {
    for (std::size_t d = 0; d < n_dets; d++)
    {
      cost(t, d) = obz::util::Dist2(
            obz::util::RectCenter(tracks_[t].GetBB()),
            obz::util::RectCenter(detections[d]));
    }
  }

  LOG(INFO) << "[MOT] cost\n" << cost;
  // Since the hungarian matcher changes the matrix in place
  cv::Mat_<int> match = cost.clone();

  hungarian_solver_.solve(match);

  LOG(INFO) << "[MOT] match\n" << match;

  CV_Assert(match.size() == cost.size());
  TICK("MOT_Hungarian");

  // Check track -> detetction tracks
  for (std::int32_t t = 0; t < match.rows; t++)
  {
    bool is_matched = false;
    std::int32_t d = 0;
    LOG(INFO) << "Checking track index " << t;
    for (d = 0; d < match.cols && !is_matched; d++)
    {
      if ((match(t, d) == 0) && (cost(t, d) < 1e4))
      {
        LOG(INFO) << t << " ****************** MATCHED AT" << d;
        is_matched = true;
        match(t, d) = -1;  // Mark detection as unmatched
      }
    }
    LOG(INFO) << "Loop ended at " << d;
    if (is_matched)
    {
      UpdateTrack(t, detections[d-1], frame, camera_transform);
      match(t, d-1) = 100;  // Matched and valid
    }
    else
    {
      UpdateTrack(t, frame, camera_transform);
    }
  }

  for (std::int32_t d = 0; d < match.cols; d++)
  {
    // Check if this detection
    std::int32_t t = 0;
    for (t = 0; t < match.rows && match(t, d) == -1; t++) ;
    if (t == match.rows)
    {
      // Create a new track
      CreateTrack(detections[d], frame);
    }
  }

  // Remove stale tracks
  LOG(INFO) << "Delete check ...";
  std::size_t num_deleted = 0;
  for (std::size_t t = 0; t < tracks_.size(); t++)
  {
    if (tracks_[t].skipped_frames > max_skipped_frames_)
    {
      DeleteTrack(t - num_deleted);
      num_deleted++;
    }
  }
  LOG(INFO) << "[MOT] Deleted Tracks: " << num_deleted;

  TICK("MOT_Tracking");
}

void MultiObjectTracker::DrawTracks(cv::Mat &frame)
{
  for (std::size_t t = 0; t < tracks_.size(); t++)
  {
    const std::uint32_t& uid = tracks_[t].uid;
    const cv::Rect& bb = tracks_[t].GetBB();
    std::stringstream text;

    const cv::Scalar track_color =
          cv::Scalar(255 * ((uid % 8) & 1), 255 * ((uid % 8) & 2), 255 * ((uid % 8) & 4));

    text << "# " << uid << " " << bb;
    cv::putText(frame, text.str(), cv::Point(bb.x, bb.y - 10), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
    cv::rectangle(frame, bb, track_color);
  }
}

} // namespace obz

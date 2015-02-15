#include "obzerver/multi_object_tracker.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/utility.hpp"

#include "glog/logging.h"

#include <cstdlib>  // for rand()
#include <sstream>
namespace obz
{

MultiObjectTracker::MultiObjectTracker(const std::uint32_t history_len,
                                       const float fps,
                                       const std::uint32_t max_skipped_frames)
  : next_uid_(1),
    history_len_(history_len),
    fps_(fps),
    max_skipped_frames_(max_skipped_frames),
    focus_track_index_(-1)
{}

MultiObjectTracker::~MultiObjectTracker()
{}

void MultiObjectTracker::CreateTrack(const cv::Rect &bb, const cv::Mat &frame)
{
  LOG(INFO) << "[MOT] Creating track " << next_uid_ << " for " << bb;

  // Create a track and append that to tracks_ vector
  tracks_.push_back(Track(next_uid_));

  // Modify the object
  Track& track = tracks_.back();

  track.object_ptr = cv::Ptr<obz::TObject>(new obz::TObject(history_len_, fps_));

  track.object_ptr->Update(bb, frame, false, false);
  track.ekf_ptr = cv::Ptr<obz::ExKalmanFilter>(new obz::ExKalmanFilter(bb));
  next_uid_++;
}

void MultiObjectTracker::DeleteTrack(const std::uint32_t track_index)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Deleting " << track_index << "'s track: " << tracks_[track_index].uid;

  if (track_index == focus_track_index_)
  {
    LOG(WARNING) << "[MOT] !!! Focus was on this track";
    focus_track_index_ = 0;
  }

  // Check dstr calls
  tracks_.erase(tracks_.begin() + track_index);
}

void MultiObjectTracker::UpdateTrack(const std::uint32_t track_index,
                                     const cv::Mat &frame,
                                     const cv::Mat &camera_transform,
                                     const bool calc_self_similarity)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Updating w/o " << track_index << "'s track: " << tracks_[track_index].uid;

  tracks_[track_index].skipped_frames++;

  // Update kalman filter w/o prediction, pass the result to TObject
  obz::object_t obj = tracks_[track_index].ekf_ptr->Update(camera_transform);
//  LOG(INFO) << "  with (pre)" << obj.bb;
  obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);
//  LOG(INFO) << "  with (post)" << obj.bb;

  tracks_[track_index].object_ptr->Update(obj,
                                          frame,
                                          calc_self_similarity,
                                          false);
}

void MultiObjectTracker::UpdateTrack(
    const std::uint32_t track_index,
    const cv::Rect &bb,
    const cv::Mat &frame,
    const cv::Mat &camera_transform,
    const bool calc_self_similarity)
{
  CV_Assert(track_index < tracks_.size());
  LOG(INFO) << "[MOT] Updating obz " << track_index << "'s track: " << tracks_[track_index].uid;
//  LOG(INFO) << "  with (pre) " << bb;

  tracks_[track_index].skipped_frames = 0;

  // Update kalman filter with bb, pass the result to TObject
  obz::object_t obj = tracks_[track_index].ekf_ptr->Update(bb, camera_transform);
  obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);
//  LOG(INFO) << "  with (post) " << obj.bb;

  tracks_[track_index].object_ptr->Update(obj,
                                          frame,
                                          calc_self_similarity,
                                          false);
}

void MultiObjectTracker::Update(const rect_vec_t &detections,
                                const cv::Mat &frame,
                                const cv::Mat& camera_transform)
{
  const std::size_t n_dets = detections.size();
  const std::size_t n_tracks = tracks_.size();

  // TDM: Choose one track at each frame to update self similarity
  const std::size_t ss_update_track_index = n_tracks ? rand() % n_tracks : 0;

  LOG(INFO) << "[MOT] Number of detections: "
            << n_dets
            << " Tracks: "
            << n_tracks
            << " SS Track: " << ss_update_track_index;

  if (n_dets == 0)
  {
//    LOG(INFO) << "[MOT] No detections";
    for (std::size_t t = 0; t < n_tracks; t++)
    {
      UpdateTrack(t, frame, camera_transform, (ss_update_track_index == t));
    }
    return;
  }

  if (n_tracks == 0)
  {
//    LOG(INFO) << "[MOT] No tracks";
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

//  LOG(INFO) << "[MOT] cost\n" << cost;
  // Since the hungarian matcher changes the matrix in place
  cv::Mat_<int> match = cost.clone();

  // It seems as if the hungarian solver class keeps state,
  // TODO: Write a reset function
  obz::alg::Hungarian hungarian_solver_;
  hungarian_solver_.diag(false);
  hungarian_solver_.solve(match);

//  LOG(INFO) << "[MOT] match\n" << match;

  CV_Assert(match.size() == cost.size());
  TICK("MOT_Hungarian");

  // Check track -> detetction tracks
  for (std::int32_t t = 0; t < match.rows; t++)
  {
    bool is_matched = false;
    std::int32_t d = 0;
//    LOG(INFO) << "Checking track index " << t;
    for (d = 0; d < match.cols && !is_matched; d++)
    {
      if ((match(t, d) == 0) && (cost(t, d) < 1e4))
      {
        is_matched = true;
        match(t, d) = -1;  // Mark detection as unmatched
      }
    }
    if (is_matched)
    {
      UpdateTrack(t, detections[d-1], frame, camera_transform,
          static_cast<std::int32_t>(ss_update_track_index) == t);
      match(t, d-1) = 100;  // Matched and valid
    }
    else
    {
      UpdateTrack(t, frame, camera_transform,
                  static_cast<std::int32_t>(ss_update_track_index) == t);
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
  std::size_t num_deleted = 0;
  for (std::size_t t = 0; t < tracks_.size(); t++)
  {
    if (tracks_[t].skipped_frames > max_skipped_frames_)
    {
      DeleteTrack(t - num_deleted);
      num_deleted++;
    }
  }

  // Update focus

  if (focus_track_index_ == -1)
  {
    if (tracks_.size())
    {
      focus_track_index_ = rand() % tracks_.size();
      tracks_[focus_track_index_].object_ptr->Reset();
      LOG(INFO) << "[MOT] Set focus on " << tracks_[focus_track_index_].uid;
    }
  }

  if (focus_track_index_ != -1)
  {
    LOG(INFO) << "[MOT] Focus is on " << tracks_[focus_track_index_].uid;
    if (tracks_[focus_track_index_].object_ptr->GetSelfSimilarity()->IsFull())
    {
      if (tracks_[focus_track_index_].object_ptr->GetPeriodicity().GetDominantFrequency(1) == -1)
      {
        LOG(INFO) << "[MOT] Non-periodic. Reseting";
        focus_track_index_ = -1;
      } else if (tracks_[focus_track_index_].object_ptr->GetPeriodicity().GetDominantFrequency(1) >= 0.75)
      {
        LOG(INFO) << "[MOT] FOUND A PERIODIC TRACK";
      }
    }
  }
//  LOG(INFO) << "[MOT] Deleted Tracks: " << num_deleted;

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

    text << "# " << uid
         << " " << tracks_[t].object_ptr->GetPeriodicity().GetDominantFrequency();
//         << " " << bb;

    cv::putText(frame, text.str(), cv::Point(bb.x, bb.y - 10), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
    cv::rectangle(frame, bb, track_color);

    cv::Mat ss_mat = tracks_[t].object_ptr->GetSelfSimilarity().GetSimMatrixRendered();
    const cv::Rect roi = util::ClampRect(cv::Rect(bb.x, bb.y, ss_mat.cols, ss_mat.rows),
                                         frame.cols, frame.rows);
    cv::cvtColor(ss_mat, ss_mat, CV_GRAY2BGR);
//    if (roi.area())
//    {
//      ss_mat(cv::Rect(0, 0, roi.width, roi.height)).copyTo(frame(roi));
//    }
    //frame(roi) = ss_mat(cv::Rect(0, 0, roi.width, roi.height)).clone();
    //cv::rectangle(frame, roi, cv::Scalar(100,100,100));

    text.str("");
    text << "./sim/track-" << tracks_[t].uid << "-ss.png";
    cv::imwrite(text.str(), ss_mat);

    /* TEMP */
    cv::Mat ss_mat_copy, ss_auto_cc;
    cv::copyMakeBorder(ss_mat, ss_mat_copy, ss_mat.rows/2, ss_mat.rows/2, ss_mat.cols/2, ss_mat.cols/2, cv::BORDER_WRAP);
    cv::matchTemplate(ss_mat_copy, ss_mat, ss_auto_cc, CV_TM_CCORR_NORMED);
    double max_val = 0.0, min_val = 0.0;
    cv::minMaxLoc(ss_auto_cc, &min_val, &max_val);
    cv::Mat ac_render;
    if (max_val > 0.0) {
      ss_auto_cc.convertTo(ac_render, CV_8UC1, 255.0 / max_val);
    } else {
      ac_render = cv::Mat::zeros(ss_auto_cc.size(), CV_8UC1);
    }
    text.str("");
    text << "./sim/track-" << tracks_[t].uid << "-ss-ac.png";
    cv::imwrite(text.str(), ac_render);
  }
}

} // namespace obz

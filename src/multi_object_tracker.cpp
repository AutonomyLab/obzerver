#include "obzerver/multi_object_tracker.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/utility.hpp"

#include "glog/logging.h"

#include <cstdlib>  // for rand()
#include <sstream>
#include <memory>
#include <exception>
namespace obz
{

const std::uint32_t Track::UNKNOWN = 0;

PeriodicityWorkerThread::PeriodicityWorkerThread(MultiObjectTracker &mot) :
  mot(mot)
{
  LOG(INFO) << "[PT] cnst";
}

bool MultiObjectTracker::SetPWTTrack(std::uint32_t track_uid)
{
  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    if (focus_track_uid_ != Track::UNKNOWN) return false;
    focus_track_uid_ = track_uid;
    LOG(INFO) << "[PT] New track set to uid: " << track_uid;
  }
  pwt_condition_.notify_one();
  return true;
}

bool MultiObjectTracker::ClearPWTTrack()
{
  std::unique_lock<std::mutex> lock(pwt_mutex_);
  if (focus_track_uid_ == Track::UNKNOWN) return false;

  LOG(INFO) << "[PT] Request to clear terminate current track: "
            << tracks_[focus_track_uid_].uid;
//  clear_track_ = true;
//  pwt_condition_.wait(lock);
  LOG(INFO) << "[PT] Wait finished";
  focus_track_uid_ = Track::UNKNOWN;
  return true;
}

void MultiObjectTracker::TerminatePWT()
{
  LOG(INFO) << "[PT] Sending request to terminate the thread ";
  pwt_terminate_ = true;
  pwt_condition_.notify_all();
}

void PeriodicityWorkerThread::operator ()()
{
  // The copy operator below will resize this
  obz::mseq_t seq(0);
  float fps = 0.0;
  std::size_t hist_len = 0;
  double diff = 0.0;

  cv::Ptr<obz::SelfSimilarity> ss_ptr;
  std::unique_ptr<obz::Periodicity> per_ptr;
  while (!mot.pwt_terminate_)
  {
    {
      std::unique_lock<std::mutex> lock(mot.pwt_mutex_);
      while (!mot.pwt_terminate_ && mot.focus_track_uid_ == Track::UNKNOWN)
      {
        LOG(INFO) << "[PT] Waiting ...";
        mot.pwt_condition_.wait(lock);
      }

      if (mot.pwt_terminate_) break;
      seq = mot.tracks_[mot.focus_track_uid_].object_ptr->GetSequence();
      fps = mot.fps_;
      hist_len = mot.history_len_;
      // Safety logic check
      CV_Assert(mot.focus_track_uid_ == mot.tracks_[mot.focus_track_uid_].uid);
      LOG(INFO) << "[PT] () Track: " <<  mot.tracks_[mot.focus_track_uid_].uid
                << " Sequence Size: " << seq.size();
    }

    const long int start_t = cv::getTickCount();

    // This is thread safe
    ss_ptr = mot.tracks_[mot.focus_track_uid_].ss_ptr;

    LOG(INFO) << "[PT] () Running SS ...";
    CV_Assert(ss_ptr);

    ss_ptr->Update(seq, mot.current_time, std::to_string(mot.focus_track_uid_));

    diff = 1000.0 * ((double) (cv::getTickCount() - start_t) / (double) cv::getTickFrequency());

    if (seq.size() == hist_len)
    {
      per_ptr.reset(new obz::Periodicity(seq.size(), fps));
      LOG(INFO) << "[PT] () Calculating Periodicity " << diff << " (ms)";
      for (int i = 0; i < ss_ptr->GetSimMatrix().cols; i+=20)
      {
        // First time, reset the spectrum, then add up the power
        per_ptr->Update(ss_ptr->GetSimMatrix().row(i), i != 0, false);
        per_ptr->Update(ss_ptr->GetSimMatrix().col(i).t(), true, false);

      }

//      LOG(INFO) << "Avg Spectrum: " << cv::Mat(periodicity.GetSpectrum(), false).t();
    }
    diff = 1000.0 * ((double) (cv::getTickCount() - start_t) / (double) cv::getTickFrequency());
    LOG(INFO) << "[PT] () SS Done, updating focused track ... " << diff << " (ms)";
    {
      std::unique_lock<std::mutex> lock(mot.pwt_mutex_);

      mot.tracks_[mot.focus_track_uid_].self_similarity_rendered =
                ss_ptr->GetSimMatrixRendered().clone();

      if (seq.size() == hist_len)
      {
        CV_Assert(per_ptr);
        mot.tracks_[mot.focus_track_uid_].dom_freq =
            per_ptr->GetDominantFrequency(1);
        mot.tracks_[mot.focus_track_uid_].avg_spectrum =
            per_ptr->GetSpectrum();
      }
    }
    LOG(INFO) << "[PT] () Done. Clearing the track";
    mot.focus_track_uid_ = Track::UNKNOWN;
  }
  LOG(INFO) << "[PT] Gracefully terminated";
}

/**********************************************/
/**********************************************/
/**********************************************/
/**********************************************/
/**********************************************/

MultiObjectTracker::MultiObjectTracker(const std::uint32_t history_len,
                                       const float fps,
                                       const std::uint32_t max_skipped_frames)
  : current_time(0),
    next_uid_(Track::UNKNOWN + 1),
    history_len_(history_len),
    fps_(fps),
    max_skipped_frames_(max_skipped_frames),
    focus_order_(0),
    focus_track_uid_(Track::UNKNOWN),
    pwt_terminate_(false),
    pwt_thread_(std::thread(obz::PeriodicityWorkerThread(*this)))
{}

MultiObjectTracker::~MultiObjectTracker()
{
  LOG(INFO) << "[MOT] dstr";
  TerminatePWT();
  if (pwt_thread_.joinable())
  {
    pwt_thread_.join();
  }
  else
  {
    LOG(WARNING) << "[MOT] PWT thread was not joinable!";
  }
  LOG(INFO) << "[MOT] graceful shutdown";
}

std::uint32_t MultiObjectTracker::FindTrackIndex(const std::uint32_t track_uid)
{
  std::uint32_t track_index = 0;
  for (track_index = 0; track_index < tracks_indexes_.size(); track_index++)
  {
    if (tracks_indexes_[track_index] == track_uid) return track_index;
  }
  CV_Assert(false);
  return 0;
}

void MultiObjectTracker::CreateTrack(const cv::Rect &bb,
                                     const cv::Mat &frame,
                                     const cv::Mat &diff_frame,
                                     const float flow)
{
  LOG(INFO) << "[MOT] Creating track " << next_uid_ << " for " << bb;

  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    LOG(INFO) << "[MOT] pre tracks size: " << tracks_.size();
    // Create a track
    tracks_.insert(track_pair_t(next_uid_, Track(next_uid_)));

    // Modify the object
    Track& track = tracks_[next_uid_];
    track.object_ptr = cv::Ptr<obz::TObject>(new obz::TObject(history_len_, fps_));

    track.object_ptr->Update(bb, frame, diff_frame, false);
    track.ekf_ptr = cv::Ptr<obz::ExKalmanFilter>(new obz::ExKalmanFilter(bb));
    track.ss_ptr = cv::Ptr<obz::SelfSimilarity>(
          new obz::SelfSimilarity(history_len_, current_time));
    track.per_ptr = cv::Ptr<obz::Periodicity>(
          new obz::Periodicity(history_len_, fps_));
    LOG(INFO) << "[MOT] post tracks size: " << tracks_.size();
  }
  next_uid_++;
  LOG(INFO) << "[MOT] Next uid will be: " << next_uid_;
}

bool MultiObjectTracker::DeleteTrackIfStale(const std::uint32_t track_uid)
{
  CV_Assert(tracks_.count(track_uid) == 1);
  LOG(INFO) << "[MOT] Checking track"  << track_uid << " for deletion";

  if (track_uid == focus_track_uid_)
  {
    LOG(WARNING) << "[MOT] !!! Focus is on this track. Skipping ...";
    return false;
  }

  // Check dstr calls
  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    if (tracks_[track_uid].skipped_frames > max_skipped_frames_)
    {
      tracks_.erase(track_uid);
    }
  }
  return true;
}

void MultiObjectTracker::UpdateTrack(
    const std::uint32_t track_uid,
    const cv::Mat& frame,
    const cv::Mat& diff_frame,
    const cv::Mat& camera_transform)
{
  CV_Assert(tracks_.count(track_uid) == 1);
  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    LOG(INFO) << "[MOT] Updating w/o " << track_uid << "'s track: ";

    tracks_[track_uid].skipped_frames++;
    obz::object_t obj = tracks_[track_uid].ekf_ptr->Update(camera_transform);
    obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);

    tracks_[track_uid].object_ptr->Update(obj,
                                          frame,
                                          diff_frame,
                                          0.0,
                                          false);

    if (tracks_[track_uid].object_ptr->GetMotionHist().size() == history_len_)
    {
      std::vector<float> vec(history_len_);
      std::size_t i = 0;
      for (auto& m: tracks_[track_uid].object_ptr->GetMotionHist())
      {
        vec[i++] = m;
      }
      tracks_[track_uid].per_ptr->Update(
            vec,
            tracks_[track_uid].life != 0);
      tracks_[track_uid].life++;
    }
  }

  const std::uint32_t track_index = FindTrackIndex(track_uid);

  if (IsPWTFree() && (focus_order_ % tracks_indexes_.size() == track_index))
  {
    focus_order_++;
    SetPWTTrack(track_uid);
  }

}

void MultiObjectTracker::UpdateTrack(
    const std::uint32_t track_uid,
    const cv::Rect &bb,
    const cv::Mat &frame,
    const cv::Mat &diff_frame, const float flow,
    const cv::Mat &camera_transform)
{
  CV_Assert(tracks_.count(track_uid) == 1);
  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    LOG(INFO) << "[MOT] Updating obz " << track_uid;
    tracks_[track_uid].skipped_frames = 0;
    obz::object_t obj = tracks_[track_uid].ekf_ptr->Update(bb, camera_transform);
    obj.bb = obz::util::ClampRect(obj.bb, frame.cols, frame.rows);
    tracks_[track_uid].object_ptr->Update(obj,
                                            frame,
                                            diff_frame,
                                            flow,
                                            false);

    if (tracks_[track_uid].object_ptr->GetMotionHist().size() == history_len_)
    {
      std::vector<float> vec(history_len_);
      std::size_t i = 0;
      for (auto& m: tracks_[track_uid].object_ptr->GetMotionHist())
      {
        vec[i++] = m;
      }
      tracks_[track_uid].per_ptr->Update(
            vec,
            tracks_[track_uid].life != 0);
      tracks_[track_uid].life++;
    }
  }

  const std::uint32_t track_index = FindTrackIndex(track_uid);

  if (IsPWTFree() && (focus_order_ % tracks_indexes_.size() == track_index))
  {
    focus_order_++;
    SetPWTTrack(track_uid);
  }
}

void MultiObjectTracker::Update(const rect_vec_t& detections,
                                const std::vector<float>& flows,
                                const cv::Mat& frame,
                                const cv::Mat& diff_frame,
                                const cv::Mat& camera_transform)
{
  current_time++;

  const std::size_t n_dets = detections.size();
  std::size_t n_tracks = 0;
  const int max_cost = 200;

  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    n_tracks = tracks_.size();
    tracks_indexes_.resize(n_tracks);
    std::uint32_t i = 0;
    for (auto& track_pair: tracks_)
    {
      tracks_indexes_[i++] = track_pair.first;
    }
  }

  LOG(INFO) << "[MOT] Number of detections: "
            << n_dets
            << " Tracks: "
            << n_tracks
            << " Focus Track ID: " << focus_track_uid_;

  if (n_dets == 0)
  {
    for (auto& track_pair: tracks_)
    {
      UpdateTrack(track_pair.first, frame, diff_frame, camera_transform);
    }
    return;
  }

  if (n_tracks == 0)
  {
    // We should create tracks for all detections
    for (std::size_t d = 0; d < n_dets; d++)
    {
      CreateTrack(detections[d], frame, diff_frame, flows[d]);
    }
    return;
  }

  cv::Mat_<int> initial_cost(n_tracks, n_dets);

  // Go over all tracks and detections to calculate assignment cost
  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);

    for (std::size_t t = 0; t < n_tracks; t++)
    {
      for (std::size_t d = 0; d < n_dets; d++)
      {
        // Prevent cost(t, d) = 0
        initial_cost(t, d) = 10 + (sqrt(obz::util::Dist2(
              obz::util::RectCenter(tracks_[tracks_indexes_[t]].GetBB()),
              obz::util::RectCenter(detections[d]))));
      }
    }
  }


  LOG(INFO) << "[MOT] cost initial" << initial_cost;

  /*
   * Optimized hungarian matcher proposed in this paper:
   * Luetteke, Felix; Zhang, Xu; Franke, Joerg,
   * "Implementation of the Hungarian Method for object tracking on a
   * camera monitored transportation system,"
   * */

  // These vectors keep ids of canid/lonely(unmatched) tracks/candidates
  // Case I. All rows or cols are above max_cost
  // Case II. Individual item is greater than max_cost

  // tracks: uid, detections: index
  std::vector<std::uint32_t> candid_tracks;
  std::vector<std::uint32_t> candid_detections;

  std::vector<std::uint32_t> lonely_tracks;
  std::vector<std::uint32_t> lonely_detections;

  // Find Case I rows and cols and add their ids to lonely_* vectors
  // Indexes of Tracks/Detections which survive are added to candid_* vectors
  for (std::size_t t = 0; t < n_tracks; t++)
  {
    bool invalid_row = true;
    for (std::size_t d = 0; d < n_dets && invalid_row; d++)
    {
      if (initial_cost(t, d) < max_cost) invalid_row = false;
    }
    if (invalid_row)
    {
      lonely_tracks.push_back(tracks_indexes_[t]);
    }
    else
    {
      candid_tracks.push_back(tracks_indexes_[t]);
    }
  }

  for (std::size_t d = 0; d < n_dets; d++)
  {
    bool invalid_col = true;
    for (std::size_t t = 0; t < n_tracks && invalid_col; t++)
    {
      if (initial_cost(t, d) < max_cost) invalid_col = false;
    }
    if (invalid_col)
    {
      lonely_detections.push_back(d);
    }
    else
    {
      candid_detections.push_back(d);
    }
  }

  // Upper bound for penalty, this is for case II
  // if candid track/match pair's cost is greater than max_cost, we
  // replace that cost with max_penalty to ensure that it will never
  // end up in the global solution
  const int max_penalty = std::max(candid_tracks.size(), candid_detections.size()) * max_cost;

  LOG(INFO) << "[MOT] Candid Tracks: " << candid_tracks.size()
            << " Candid Dets: " << candid_detections.size()
            << " Max Penalty: " << max_penalty;
  cv::Mat_<int> cost(candid_tracks.size(), candid_detections.size());

  {
    std::unique_lock<std::mutex> lock(pwt_mutex_);
    for (std::size_t t = 0; t < candid_tracks.size(); t++)
    {
      for (std::size_t d = 0; d < candid_detections.size(); d++)
      {
        // Prevent cost(t, d) = 0
        const int c = 10 + (sqrt(
                              obz::util::Dist2(
                                obz::util::RectCenter(tracks_[candid_tracks[t]].GetBB()),
                                obz::util::RectCenter(detections[candid_detections[d]]))));

        // Case II
        cost(t, d) = c < max_cost ? c : max_penalty;
      }
    }
  }

  LOG(INFO) << "[MOT] candidate cost: " << cost;

  // Since the hungarian matcher changes the matrix in place
  cv::Mat_<int> match = cost.clone();

  // It seems as if the hungarian solver class keeps state,
  // TODO: Write a reset function
  obz::alg::Hungarian hungarian_solver_;
  hungarian_solver_.diag(false);
  hungarian_solver_.solve(match);

  LOG(INFO) << "[MOT] match " << match;

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
      if (match(t, d) == 0)
      {
        if (cost(t, d) != max_penalty)
        {
          is_matched = true;
        }
        else
        {
          // Push it to lonely detections if distance is max_penalty (case II)
          lonely_detections.push_back(candid_detections[d]);
        }
      }
    }
    if (is_matched)
    {
      // with obz
      UpdateTrack(candid_tracks[t],
                  detections[candid_detections[d-1]],
          frame, diff_frame, flows[candid_detections[d-1]], camera_transform);
//      match(t, d-1) = 100;  // Matched and valid
    }
    else
    {
      // Push unmatched tracks to list of tracks w/o obz
      lonely_tracks.push_back(candid_tracks[t]);
    }
  }

//  LOG(INFO) << "[MOT] match modified " << match;
  for (std::size_t d = 0; d < lonely_detections.size(); d++)
  {
    CreateTrack(detections[lonely_detections[d]], frame, diff_frame, flows[lonely_detections[d]]);
  }

  for (std::size_t t = 0; t < lonely_tracks.size(); t++)
  {
    // Update w/o obz
    UpdateTrack(lonely_tracks[t], frame, diff_frame, camera_transform);
  }

  std::size_t num_deleted = 0;
  for (std::size_t t = 0; t < tracks_indexes_.size(); t++)
  {
    num_deleted += DeleteTrackIfStale(tracks_indexes_[t]);
  }

  LOG(INFO) << "[MOT] Deleted Tracks: " << num_deleted;
  TICK("MOT_Tracking");
}

void MultiObjectTracker::DrawTracks(cv::Mat &frame)
{

  cv::Mat ss_mat;
  std::uint32_t uid;
  cv::Rect bb;
  float dom_freq;
  std::stringstream text;
  std::size_t skipped_frames;
  std::vector<float> avg_spectrum;
  std::vector<float> motion_hist;
  std::vector<float> flow_hist;

  std::unique_lock<std::mutex> lock(pwt_mutex_);
  LOG(WARNING) << "[MOT] >>>>> " << tracks_.size();
  for (auto& track_pair: tracks_)
  {
      uid = track_pair.second.uid;
      LOG(INFO) << "[MOT] " << uid << " while " << track_pair.first;
      bb = track_pair.second.GetBB();
      dom_freq = track_pair.second.dom_freq;
//      dom_freq = tracks_[t].per_ptr->GetDominantFrequency(1);
      ss_mat = track_pair.second.self_similarity_rendered.clone();
      skipped_frames = track_pair.second.skipped_frames;
      avg_spectrum = track_pair.second.avg_spectrum;
//      avg_spectrum = tracks_[t].per_ptr->GetSpectrum();
      std::size_t i = 0;
      motion_hist.resize(track_pair.second.object_ptr->GetMotionHist().size());
      for (auto& m: track_pair.second.object_ptr->GetMotionHist())
      {
        motion_hist[i++] = m;
      }

      i = 0;
      flow_hist.resize(track_pair.second.object_ptr->GetFlowHist().size());
      for (auto& f: track_pair.second.object_ptr->GetFlowHist())
      {
        flow_hist[i++] = f;
      }

    const cv::Scalar track_color =
          cv::Scalar(255 * ((uid % 8) & 1), 255 * ((uid % 8) & 2), 255 * ((uid % 8) & 4));

    text.str("");
    text << "# " << uid
         << " " << dom_freq;
//         << " " << bb;

    cv::putText(frame, text.str(),
                cv::Point(bb.x, bb.y - 10),
                CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
    cv::rectangle(frame, bb, track_color);

    LOG(INFO) << "[MOT] Track uid: " << uid << " " << bb << " " << dom_freq
                 << " SF " << skipped_frames;
    if (motion_hist.size())
    {
      LOG(INFO) << "Motion hist: " << cv::Mat(motion_hist, false).t();
    }
    if (flow_hist.size())
    {
      LOG(INFO) << "Flow hist: " << cv::Mat(flow_hist, false).t();
    }


    if (avg_spectrum.size())
    {
      LOG(INFO) << "Avg Spec: " << cv::Mat(avg_spectrum, false).t();
    }

    if (ss_mat.data)
    {
      const cv::Rect roi = util::ClampRect(cv::Rect(bb.x, bb.y, ss_mat.cols, ss_mat.rows),
                                           frame.cols, frame.rows);

      cv::cvtColor(ss_mat, ss_mat, CV_GRAY2BGR);
      if (roi.area())
      {
        cv::Mat ss_mat_roi = ss_mat(cv::Rect(0, 0, roi.width, roi.height)).mul(0.8);
        cv::Mat frame_roi_orig = frame(roi).clone().mul(0.2);
        ss_mat_roi += frame_roi_orig;
        ss_mat_roi.copyTo(frame(roi));
      }
    }

//    text.str("");
//    text << "./sim/track-" << tracks_[t].uid << "-ss.png";
//    cv::imwrite(text.str(), ss_mat);

    /* TEMP */
//    cv::Mat ss_mat_copy, ss_auto_cc;
//    cv::copyMakeBorder(ss_mat, ss_mat_copy, ss_mat.rows/2, ss_mat.rows/2, ss_mat.cols/2, ss_mat.cols/2, cv::BORDER_WRAP);
//    cv::matchTemplate(ss_mat_copy, ss_mat, ss_auto_cc, CV_TM_CCORR_NORMED);
//    double max_val = 0.0, min_val = 0.0;
//    cv::minMaxLoc(ss_auto_cc, &min_val, &max_val);
//    cv::Mat ac_render;
//    if (max_val > 0.0) {
//      ss_auto_cc.convertTo(ac_render, CV_8UC1, 255.0 / max_val);
//    } else {
//      ac_render = cv::Mat::zeros(ss_auto_cc.size(), CV_8UC1);
//    }
//    text.str("");
//    text << "./sim/track-" << tracks_[t].uid << "-ss-ac.png";
//    cv::imwrite(text.str(), ac_render);
  }
}

} // namespace obz

#ifndef PERIODICITY_APP_H
#define PERIODICITY_APP_H

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <opencv2/features2d.hpp>

#include "obzerver/common_types.hpp"
#include "obzerver/logger.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"
#include "obzerver/roi_extraction.hpp"
#include "obzerver/multi_object_tracker.hpp"

namespace obz
{
namespace app
{

namespace po = boost::program_options;

class PeriodicityApp
{
private:
  bool inited_;
  // Boost Program Options
  po::options_description po_config_options;
  bool eval_mode_;
  std::string eval_filename_;

  // Config
  std::string cascade_src;
  float param_fps;
  int param_ffd_threshold;
  std::size_t param_max_features;
  std::size_t param_hist_len;
  std::size_t param_pylk_winsize;
  std::size_t param_pylk_iters;
  double param_pylk_eps;
  double param_dbs_eps;
  std::size_t param_dbs_min_elements;
  std::size_t param_dbs_threads;
  float param_roi_min_motion_ppx;
  float param_roi_min_motion_pft;
  float param_roi_min_flow_ppx;
  float param_roi_inflation_width;
  float param_roi_inflation_height;
  std::uint32_t param_roi_min_height;
  std::uint32_t param_roi_min_width;
  std::uint32_t param_roi_max_height;
  std::uint32_t param_roi_max_width;
  float eval_decision_f_low;
  float eval_decision_f_high;
  std::size_t eval_min_hit_frames;
  std::uint32_t param_mot_method;
  std::size_t param_mot_max_skipped_frames;
  float param_mot_max_matching_cost;

  // Variables
  cv::Mat frame_gray;
  cv::Mat debug_frame;
  cv::Mat stab_frame;
  std::size_t eval_frame_counter;
  std::size_t eval_uid;
  bool eval_done_;

  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<obz::CameraTracker> camera_tracker;
  cv::Ptr<obz::ROIExtraction> roi_extraction;
  cv::Ptr<obz::MultiObjectTracker> multi_object_tracker;

  obz::rect_vec_t rois;
  std::vector<float> flows;
  std::vector<obz::Track> tracks;
  std::map<std::size_t, std::size_t> periodic_uids_map;

public:
  PeriodicityApp();
  ~PeriodicityApp();

  po::options_description& GetOptionsDescription() {return po_config_options;}

  bool Init(const std::string& config_filename,
            const std::string& log_filename,
            const bool eval_mode,
            const std::string& eval_filename,
            po::variables_map& boots_po_vm);

  std::size_t Update(const cv::Mat& frame);

  bool Alive() const {return !(eval_mode_ && eval_done_); }

  std::size_t GetPeriodicTracks(std::vector<Track> &per_tracks) const;

  const cv::Ptr<cv::FeatureDetector>& GetFDCstPtr() const {return feature_detector;}
  const cv::Ptr<obz::CameraTracker>& GetCTCstPtr() const {return camera_tracker;}
  const cv::Ptr<obz::ROIExtraction>& GetRECstPtr() const {return roi_extraction;}
  cv::Ptr<obz::MultiObjectTracker>& GetMOTPtr() {return multi_object_tracker;}

};

}  // namespace app
}  // namespace obz
#endif

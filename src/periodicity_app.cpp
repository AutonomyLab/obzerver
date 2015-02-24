#include <fstream>

#include "obzerver/periodicity_app.hpp"

namespace obz
{
namespace app
{

PeriodicityApp::PeriodicityApp() :
  inited_(false),
  po_config_options("PeriodicityApp Configuration"),
  eval_frame_counter(0),
  eval_uid(0),
  eval_done_(false)
{
  po_config_options.add_options()
      ("history,hi", po::value<std::size_t>(&param_hist_len)->default_value(120), "Length of history (frames)")
      ("fps", po::value<float>(&param_fps)->default_value(30.0), "frames per second")
      ("eval.f_low", po::value<float>(&eval_decision_f_low)->default_value(0.9), "Decision min frequency")
      ("eval.f_high", po::value<float>(&eval_decision_f_high)->default_value(3.1), "Decision max frequency")
      ("eval.min_frames", po::value<std::size_t>(&eval_min_hit_frames)->default_value(5), "Minimum number of consequtive positive hits before making the decision")
      ("icf.cascade", po::value<std::string>(&cascade_src), "icf cascade file")
      ("stablize.numfeatures", po::value<std::size_t>(&param_max_features)->default_value(300), "Number of features to track for stablization")
      ("stablize.ffd_threshold", po::value<int>(&param_ffd_threshold)->default_value(30), "Fast Feature Detector threshold")
      ("stablize.pylk_winsize", po::value<std::size_t>(&param_pylk_winsize)->default_value(30), "Size of search window size for pylk")
      ("stablize.pylk_iters", po::value<std::size_t>(&param_pylk_iters)->default_value(30), "Number of iterations for pylk")
      ("stablize.pylk_eps", po::value<double>(&param_pylk_eps)->default_value(0.01), "pylk eps criteria")
      ("dbscan.eps", po::value<double>(&param_dbs_eps)->default_value(0.04), "DBScan Threshold (0,1) ")
      ("dbscan.min_elements", po::value<std::size_t>(&param_dbs_min_elements)->default_value(10), "in number of cluster members")
      ("dbscan.threads", po::value<std::size_t>(&param_dbs_threads)->default_value(2), "DBScan OpenMP Threads")
      ("roi.min_motion_ppx", po::value<float>(&param_roi_min_motion_ppx)->default_value(0.01), "Min sum(diff(roi))/roi.size() to accept the ROI")
      ("roi.min_motion_pft", po::value<float>(&param_roi_min_motion_pft)->default_value(40), "Min diff value for a feature point to be considered for clustering")
      ("roi.min_flow_ppx", po::value<float>(&param_roi_min_flow_ppx)->default_value(0.1), "Min sum(|flow(roi)|/roi.size() to accept the ROI")
      ("roi.inflation_width", po::value<float>(&param_roi_inflation_width)->default_value(0.75), "How much to inflate the width of and extracted and accepted ROI (0.5: 0.25 increase for each side")
      ("roi.inflation_height", po::value<float>(&param_roi_inflation_height)->default_value(0.5), "How much to inflate the height of an extracted and accepted ROI (0.5: 0.25 increase for each side")
      ("mot.method", po::value<std::uint32_t>(&param_mot_method)->default_value(0), "Periodicity Detection Method 0: SelfSimilarity 1: Average Diff Motion")
      ("mot.max_skipped_frames", po::value<std::size_t>(&param_mot_max_skipped_frames)->default_value(30), "Maximum number of non-matching obzervation before deleting a track")
      ("mot.max_matching_cost", po::value<float>(&param_mot_max_matching_cost)->default_value(100), "Maximum tolerable eucledian distance when mathcing tracks and observations (in pixels)")
      //      ("", po::value<>()->default_value(), "")
      ;
}

PeriodicityApp::~PeriodicityApp()
{}

bool PeriodicityApp::Init(const std::string& config_filename,
                          const std::string& log_filename,
                          const bool eval_mode,
                          const std::string& eval_filename,
                          po::variables_map& boots_po_vm)
{
  if (!config_filename.empty())
  {
    std::ifstream config_file(config_filename);
    if (config_file)
    {
      po::store(po::parse_config_file(config_file, po_config_options), boots_po_vm);
      po::notify(boots_po_vm);
    }
    else
    {
      std::cerr << "Can not open the configuration file " << config_file << std::endl;
      return false;
    }
  }
  eval_mode_ = eval_mode;
  eval_filename_ = eval_filename;
  obz::log_config("papp", log_filename);
  inited_ = true;

  feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);
  camera_tracker = new obz::CameraTracker(
        param_hist_len,
        feature_detector,
        param_max_features,
        param_pylk_winsize,
        param_pylk_iters,
        param_pylk_eps);

  roi_extraction = new obz::ROIExtraction(
        param_dbs_eps,
        param_dbs_min_elements,
        cv::Size(5, 10),
        cv::Size(100, 200),
        param_roi_min_motion_ppx,  // Min Avg Motion Per Pixel
        param_roi_min_motion_pft,  //
        param_roi_min_flow_ppx,  // Min Avg Optical Flow Per Pixel
        param_roi_inflation_width,  // Inflation: Width
        param_roi_inflation_height,  // Inflation: Height
        param_dbs_threads);   // Num Threads

  multi_object_tracker = new obz::MultiObjectTracker(
        (param_mot_method == 0) ?
          obz::PERIODICITY_SELFSIMILARITY:
          obz::PERIODICITY_AVERAGEMOTION,
        param_hist_len,
        param_fps,
        param_mot_max_skipped_frames,
        param_mot_max_matching_cost);
  return true;
}

std::size_t PeriodicityApp::Update(const cv::Mat &frame)
{
  CV_Assert(inited_);
  cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
  const bool ct_success = camera_tracker->Update(frame_gray, frame);

  std::size_t num_tracks = 0;
  if (!ct_success) {
    LOG(WARNING) << "[PA] Camera Tracker Failed";
  } else {
    roi_extraction->Update(camera_tracker->GetTrackedFeaturesCurr(),
                          camera_tracker->GetTrackedFeaturesPrev(),
                          camera_tracker->GetLatestDiff());

    rois.clear();
    flows.clear();
    tracks.clear();

    roi_extraction->GetValidBBs(rois, flows);
    multi_object_tracker->Update(rois,
                                flows,
                                camera_tracker->GetStablizedGray(),
                                camera_tracker->GetLatestDiff(),
                                camera_tracker->GetLatestCameraTransform().inv(),
                                camera_tracker->GetFullLengthCameraTransform().inv());

    num_tracks = multi_object_tracker->CopyAllTracks(tracks);

    // uid -> index
    periodic_uids_map.clear();
    for (std::size_t i = 0; i < num_tracks; i++)
    {
      const obz::Track& tr = tracks[i];
      LOG(INFO) << "[PA] uid: " << tr.uid
                << " bb: " << tr.GetBB()
                << " freq: " << tr.dom_freq
                << " disp: " << tr.displacement;

      if ((tr.dom_freq >= eval_decision_f_low) &&
          (tr.dom_freq < eval_decision_f_high))
      {
        periodic_uids_map.insert(std::pair<std::size_t, std::size_t>(tr.uid, i));
      }
    }

    LOG(INFO) << "[PA] Number of periodic tracks: " << periodic_uids_map.size();

    if (eval_mode_)
    {
      if (eval_uid)
      {
        if (periodic_uids_map.count(eval_uid))
        {
          eval_frame_counter++;
        }
        else
        {
          eval_uid = 0;
          eval_frame_counter = 0;
        }
      }
      else
      {
        if (periodic_uids_map.size())
        {
          // TODO: What if there are more than one periodic tracks?
          eval_uid = (*(periodic_uids_map.begin())).first;
          eval_frame_counter = 1;
        }
      }
    }

    if (eval_frame_counter > eval_min_hit_frames)
    {
      const obz::Track& tr = tracks[periodic_uids_map[eval_uid]];
      LOG(INFO) << "[PA] Decision Made: " << eval_filename_;
      cv::imwrite(eval_filename_, camera_tracker->GetStablizedRGB()(tr.GetBB()).clone());
      std::ofstream metadata_file(eval_filename_ + std::string(".txt"), std::ios::out);
      metadata_file << tr.uid << std::endl;
      metadata_file << tr.GetBB() << std::endl;
      metadata_file << tr.dom_freq << std::endl;
      metadata_file << cv::Mat(tr.avg_spectrum, false).t() << std::endl;
      metadata_file.close();
      eval_done_ = true;
    }
  }
  return num_tracks;
}

std::size_t PeriodicityApp::GetPeriodicTracks(std::vector<Track> &per_tracks) const
{
  per_tracks.resize(periodic_uids_map.size());
  std::size_t i = 0;
  for (auto& uid_index: periodic_uids_map)
  {
    per_tracks[i] = tracks[uid_index.second];
    i++;
  }
  return i;
}

}  // namespace app
}  // namespace obz

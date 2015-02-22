#include <iostream>
#include <iomanip>
#include <fstream>

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "obzerver/common_types.hpp"
#include "obzerver/logger.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"
//#include "obzerver/object_tracker.hpp"
#include "obzerver/roi_extraction.hpp"
#include "obzerver/multi_object_tracker.hpp"

void mouseCallback(int event, int x, int y, int flags, void* data) {
  (void) flags;  // shutup gcc
  if (event == cv::EVENT_LBUTTONDOWN) {
    bool* pause = (bool*) data;
    *pause = !(*pause);
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    std::cout << "Right click at " << x << " , " << y << std::endl;
  }
}

namespace po = boost::program_options;
int main(int argc, char* argv[])
{
  /* Params and Command Line */

  std::string video_src;
  std::string config_filename;
  std::string cascade_src;
  std::string logfile;

  bool display;
  bool clear;
  bool viz_features;
  bool viz_rois;
  bool viz_tracks;
  bool pause;
  bool loop;
  std::size_t start_frame;

  float param_downsample_factor;
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

  bool eval_mode;
  float eval_decision_f_low;
  float eval_decision_f_high;
  std::string eval_file;

  std::uint32_t param_mot_method;
  std::size_t param_mot_max_skipped_frames;
  float param_mot_max_matching_cost;

  // This will be merged with config file options
  po::options_description po_generic_desc("Generic Options");
  po_generic_desc.add_options()
      ("help,h", "Show help")
      ("video,v", po::value<std::string>(&video_src), "Video file")
      ("config,c", po::value<std::string>(&config_filename)->default_value(""), "Configuration File (INI)")
      ("display,d", po::bool_switch(&display)->default_value(false), "Show visualization")
      ("viz.features", po::bool_switch(&viz_features)->default_value(false), "Visualize Features")
      ("viz.rois", po::bool_switch(&viz_rois)->default_value(false), "Visualize ROIS")
      ("viz.tracks", po::bool_switch(&viz_tracks)->default_value(false), "Visualize Tracks")
      ("clear,cl", po::bool_switch(&clear)->default_value(false), "Clear Terminal")
      ("pause,p", po::bool_switch(&pause)->default_value(false), "Start in pause mode")
      ("logfile,l", po::value<std::string>(&logfile)->default_value(""), "specify log file (empty: log to stderr)")
      ("skip,k", po::value<std::size_t>(&start_frame)->default_value(0), "Starting frame")
      ("eval.enabled,e", po::bool_switch(&eval_mode)->default_value(false), "Evaluation (Experiment/Decistion) mode")
      ("eval.file", po::value<std::string>(&eval_file)->default_value("/tmp/obzerver.png"), "Image file to dump the decision bounding box to")
      ("loop", po::bool_switch(&loop)->default_value(false), "loop video")
      ;

  // Config file
  po::options_description po_config_options("Configuration");
  po_config_options.add_options()
      ("history,hi", po::value<std::size_t>(&param_hist_len)->default_value(120), "Length of history (frames)")
      ("fps", po::value<float>(&param_fps)->default_value(30.0), "frames per second")
      ("downsample", po::value<float>(&param_downsample_factor)->default_value(1.0), "Downsample (resize) factor (0.5: half)")
      ("eval.f_low", po::value<float>(&eval_decision_f_low)->default_value(0.9), "Decision min frequency")
      ("eval.f_high", po::value<float>(&eval_decision_f_high)->default_value(3.1), "Decision max frequency")
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

  // Generic and Config file options
  po::options_description po_cmdline_options;
  po_cmdline_options.add(po_generic_desc).add(po_config_options);

  // Parse all first
  po::variables_map po_vm;
  po::store(po::parse_command_line(argc, argv, po_cmdline_options), po_vm);
  po::notify(po_vm);

  if (!config_filename.empty())
  {
    std::ifstream config_file(config_filename);
    if (config_file)
    {
      po::store(po::parse_config_file(config_file, po_config_options), po_vm);
      po::notify(po_vm);
    }
    else
    {
      std::cerr << "Can not open the configuration file " << config_file << std::endl;
      return 1;
    }
  }

  if (po_vm.count("help") || video_src.empty())
  {
    std::cout << po_cmdline_options << std::endl;
  }

  /* Logger */

  obz::log_config(argv[0], logfile);

  /* Initial Logging */
  LOG(INFO) << "Video Source: " << video_src;
  LOG(INFO) << "Logfile: " << logfile;
  if (!config_filename.empty())
  {
    LOG(INFO) << "Config file: " << config_filename;
    LOG(INFO) << "---------------------------------";
    std::ifstream config_file(config_filename);
    std::string line;
    while (std::getline(config_file, line))
    {
      LOG(INFO) << line;
    }
    LOG(INFO) << "---------------------------------";
  }

  /* Variables */
  const bool use_webcam  = (video_src.compare("cam") == 0);
  unsigned long int frame_counter = 0;
  cv::Mat frame;
  cv::Mat frame_gray;
  cv::Mat debug_frame;
  cv::Mat stab_frame;

  StepBenchmarker& ticker = StepBenchmarker::GetInstance();
  cv::VideoCapture capture;
  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);

//  cv::Ptr<ccv::ICFCascadeClassifier> ccv_icf_ptr = 0;
//    cv::Ptr<cv::FeatureDetector> feature_detector = new cv::BRISK(param_ffd_threshold);
//    cv::Ptr<cv::FeatureDetector> feature_detector = new cv::GoodFeaturesToTrackDetector(param_max_features);

  obz::CameraTracker camera_tracker(param_hist_len,
                                    feature_detector,
                                    param_max_features,
                                    param_pylk_winsize,
                                    param_pylk_iters,
                                    param_pylk_eps);

  obz::ROIExtraction roi_extraction(param_dbs_eps,
                                    param_dbs_min_elements,
                                    cv::Size(5, 10),
                                    cv::Size(100, 200),
                                    param_roi_min_motion_ppx,  // Min Avg Motion Per Pixel
                                    param_roi_min_motion_pft,  //
                                    param_roi_min_flow_ppx,  // Min Avg Optical Flow Per Pixel
                                    param_roi_inflation_width,  // Inflation: Width
                                    param_roi_inflation_height,  // Inflation: Height
                                    param_dbs_threads);   // Num Threads


  obz::MultiObjectTracker multi_object_tracker(
        (param_mot_method == 0) ?
          obz::PERIODICITY_SELFSIMILARITY:
          obz::PERIODICITY_AVERAGEMOTION,
        param_hist_len,
        param_fps,
        param_mot_max_skipped_frames,
        param_mot_max_matching_cost);

  obz::util::trackbar_data_t trackbar_data(&capture, &frame_counter);  

  int opengl_flags = 0;
  if (display)
  {
    try {
      cv::namedWindow("dummy", cv::WINDOW_OPENGL);
      opengl_flags = cv::WINDOW_OPENGL;
      cv::destroyWindow("dummy");
    } catch (const cv::Exception& ex) {
      LOG(WARNING) << "OpenCV without OpenGL support.";
    }
  }


  std::size_t num_frames = 0;

  try {
    if (use_webcam && !capture.open(0)) {
      LOG(ERROR) << "Can not webcam";
      return 1;
    } else if (!use_webcam && !capture.open(video_src)) {
      LOG(ERROR) << "Can not open file: " << video_src;
      return 1;
    }
    num_frames = use_webcam ? 0 : capture.get(CV_CAP_PROP_FRAME_COUNT);
    if (!use_webcam && start_frame > 0 && start_frame < num_frames) {
      frame_counter = start_frame;
      capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
    }
    if (!use_webcam) {
      LOG(INFO) << "Openning file: " << video_src << " frames: " << capture.get(CV_CAP_PROP_FRAME_COUNT);
    }
    if (display) {      
      cv::namedWindow("Original", cv::WINDOW_AUTOSIZE | opengl_flags);
      cv::namedWindow("DiffStab", cv::WINDOW_NORMAL | opengl_flags);
      cv::namedWindow("Debug", cv::WINDOW_NORMAL | opengl_flags);
      if (num_frames > 0) {
        cv::createTrackbar("Browse", "Original", 0, num_frames, obz::util::trackbarCallback, &trackbar_data);
        cv::setTrackbarPos("Browse", "Original", frame_counter);
      }
      cv::setMouseCallback("Original", mouseCallback, (void*) &pause);
    }

    ticker.reset();
    while (capture.read(frame)) {
      ticker.tick("ML_Frame_Capture");
      if (param_downsample_factor < 1.0 && param_downsample_factor > 0.0) {
        cv::resize(frame, frame, cv::Size(0, 0), param_downsample_factor, param_downsample_factor, cv::INTER_CUBIC);
        ticker.tick("ML_Downsampling");
      }
      LOG(INFO) << "Frame: " << frame_counter << " [" << frame.cols << " x " << frame.rows << "]";
      if (display && !use_webcam && (frame_counter % 10 == 0)) cv::setTrackbarPos("Browse", "Original", frame_counter);
      cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
//      cv::equalizeHist(frame_gray, frame_gray);
      ticker.tick("ML_Frame_2_Gray");
      bool ct_success = camera_tracker.Update(frame_gray, frame);
//      cv::Point2d center;
//      double _w=0.0, _h=0.0, _f=-1.0;
      if (!ct_success) {
        LOG(WARNING) << "Camera Tracker Failed";
        // TODO
      } else {
        roi_extraction.Update(camera_tracker.GetTrackedFeaturesCurr(),
                              camera_tracker.GetTrackedFeaturesPrev(),
                              camera_tracker.GetLatestDiff());
        obz::rect_vec_t rois;
        std::vector<float> flows;
        roi_extraction.GetValidBBs(rois, flows);
        multi_object_tracker.Update(rois,
                                    flows,
                                    camera_tracker.GetStablizedGray(),
                                    camera_tracker.GetLatestDiff(),
                                    camera_tracker.GetLatestCameraTransform().inv());

//        LOG(INFO) << "Tracking status: " << object_tracker.GetStatus();
//        if (object_tracker.IsTracking()) {
//          _f = object_tracker.GetObject().GetPeriodicity().GetDominantFrequency(1); // TODO
//          LOG(INFO) << "Object: "
//                    << object_tracker.GetObjectBoundingBox()
//                    << " Periodicity:"
//                    << _f;
//          //LOG(INFO) << "Spectrum: " << cv::Mat(object_tracker.GetObject().GetPeriodicity().GetSpectrum(), false);
//          if (eval_mode && _f >= decision_f_low && _f <= decision_f_high)
//          {
//            cv::imwrite(eval_file,
//                        camera_tracker.GetStablizedGray()(
//                          object_tracker.GetObjectBoundingBox()).clone());
//            LOG(ERROR) << "######## DECISION MADE : " << _f;
//            break;
//          }
//        }

      }

      if (display) {

        cv::Mat diff_frame = camera_tracker.GetLatestDiff();
        diff_frame.convertTo(diff_frame, CV_8UC1, 5.0);
        cv::Mat debug_frame = camera_tracker.GetStablizedGray();
        cv::cvtColor(debug_frame, debug_frame, CV_GRAY2BGR);

        if (viz_tracks) multi_object_tracker.DrawTracks(frame);

        if (viz_features && camera_tracker.GetTrackedFeaturesCurr().size()) {
          obz::util::DrawFeaturePointsTrajectory(debug_frame,
                                      camera_tracker.GetHomographyOutliers(),
                                      camera_tracker.GetTrackedFeaturesPrev(),
                                      camera_tracker.GetTrackedFeaturesCurr(),
                                      2,
                                      CV_RGB(0,0,255), CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));
        }

        if (viz_rois) roi_extraction.DrawROIs(debug_frame, true);

//        cv::rectangle(diff_frame, cv::Rect(center.x - _w/2, center.y-_h/2, _w, _h), CV_RGB(255, 255, 255));

//        std::stringstream ss;
//        ss << std::setprecision(5) << "Periodicity: " << _f;
//        cv::putText(frame, ss.str(), cv::Point(40,40), 1, CV_FONT_HERSHEY_PLAIN, cv::Scalar(255, 0, 0));
//        //cv::circle(diff_frame, center, 10, CV_RGB(255, 255, 255));
        if (frame.data) cv::imshow("Original", frame);
        if (diff_frame.data) cv::imshow("DiffStab", diff_frame);
        if (debug_frame.data) cv::imshow("Debug", debug_frame);
        cv::waitKey(5);
        ticker.tick("ML_Visualization");
      }
      LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
      //ticker.dump(clear);
      while (display && pause) cv::waitKey(100);
      frame_counter++;
      LOG(INFO) << frame_counter << " / " << num_frames;
      if (!use_webcam && loop && frame_counter == num_frames-1)
      {
        frame_counter = start_frame;
        capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
      }
      ticker.reset();
    }
  } catch (const std::runtime_error& e) {
//  } catch (const cv::Exception& ex) {
//    LOG(ERROR) << "Exception: " << ex.what();
//    if (capture.isOpened()) capture.release();
//    LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
    return 1;
  }
  if (capture.isOpened()) capture.release();
  LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
  return 0;
}

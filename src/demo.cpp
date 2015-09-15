#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "obzerver/periodicity_app.hpp"

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
  std::string logfile;

  float param_downsample_factor;
  bool display;
  bool clear;
  bool viz_features;
  bool viz_rois;
  bool viz_tracks;
  bool pause;
  bool loop;
  bool eval_mode;
  std::string eval_file;
  std::size_t start_frame;

  obz::app::PeriodicityApp papp;

  // This will be merged with config file options
  po::options_description po_generic_desc("Generic Options");
  po_generic_desc.add_options()
      ("help,h", "Show help")
      ("video,v", po::value<std::string>(&video_src), "Video file")
      ("downsample", po::value<float>(&param_downsample_factor)->default_value(1.0), "Downsample (resize) factor (0.5: half)")
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


  // Generic and App options
  po::options_description po_cmdline_options;
  po_cmdline_options.add(po_generic_desc).add(papp.GetOptionsDescription());

  // Parse all first
  po::variables_map po_vm;
  po::store(po::parse_command_line(argc, argv, po_cmdline_options), po_vm);
  po::notify(po_vm);

  if (po_vm.count("help") || video_src.empty())
  {
    std::cout << po_cmdline_options << std::endl;
    return 2;
  }

  // Initialize the App and the Logger
  if (!papp.Init(config_filename, logfile, eval_mode, eval_file, po_vm))
  {
    std::cerr << "Fatal error when initializing the app" << std::endl;
    return 1;
  }

  /* Initial Logging */
  LOG(INFO) << "[ML] Video Source: " << video_src;
  LOG(INFO) << "[ML] Logfile: " << logfile;
  if (!config_filename.empty())
  {
    LOG(INFO) << "[ML] Config file: " << config_filename;
    LOG(INFO) << "[ML] ---------------------------------";
    std::ifstream config_file(config_filename);
    std::string line;
    while (std::getline(config_file, line))
    {
      LOG(INFO) << "[ML] " << line;
    }
    LOG(INFO) << "[ML] ---------------------------------";
  }

  /* Variables */
  unsigned long int frame_counter = 0;
  std::size_t num_frames = 0;
  const bool use_webcam  = (video_src.compare("cam") == 0);
  cv::Mat frame;
  cv::VideoCapture capture;
  obz::util::trackbar_data_t trackbar_data(&capture, &frame_counter);  

  int opengl_flags = 0;
  if (display)
  {
    try {
      cv::namedWindow("dummy", cv::WINDOW_OPENGL);
      opengl_flags = cv::WINDOW_OPENGL;
      cv::destroyWindow("dummy");
    } catch (const cv::Exception& ex) {
      LOG(WARNING) << "[ML] OpenCV without OpenGL support.";
    }
  }

  try {
    if (use_webcam && !capture.open(0)) {
      LOG(ERROR) << "[ML] Can not webcam";
      return 1;
    } else if (!use_webcam && !capture.open(video_src)) {
      LOG(ERROR) << "[ML] Can not open file: " << video_src;
      return 1;
    }
    num_frames = use_webcam ? 0 : capture.get(CV_CAP_PROP_FRAME_COUNT);
    if (!use_webcam && start_frame > 0 && start_frame < num_frames) {
      frame_counter = start_frame;
      capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
    }
    if (!use_webcam) {
      LOG(INFO) << "[ML] Openning file: " << video_src << " frames: " << capture.get(CV_CAP_PROP_FRAME_COUNT);
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

    StepBenchmarker::GetInstance().reset();
    while (capture.read(frame) && papp.Alive()) {
      TICK("ML_Frame_Capture");
      if (param_downsample_factor < 1.0 && param_downsample_factor > 0.0) {
        cv::resize(frame, frame, cv::Size(0, 0), param_downsample_factor, param_downsample_factor, cv::INTER_CUBIC);
        TICK("ML_Downsampling");
      }
      LOG(INFO) << "[ML] Frame: " << frame_counter << " [" << frame.cols << " x " << frame.rows << "]";
      if (display && !use_webcam && (frame_counter % 10 == 0)) cv::setTrackbarPos("Browse", "Original", frame_counter);

      papp.Update(frame);

      if (display) {

        cv::Mat diff_frame = papp.GetCTCstPtr()->GetLatestDiff();
        diff_frame.convertTo(diff_frame, CV_8UC1, 5.0);
        cv::Mat debug_frame = papp.GetCTCstPtr()->GetStablizedGray();
        cv::cvtColor(debug_frame, debug_frame, CV_GRAY2BGR);

        if (viz_tracks) papp.GetMOTPtr()->DrawTracks(frame);

        if (viz_features && papp.GetCTCstPtr()->GetTrackedFeaturesCurr().size()) {
          obz::util::DrawFeaturePointsTrajectory(debug_frame,
                                      papp.GetCTCstPtr()->GetHomographyOutliers(),
                                      papp.GetCTCstPtr()->GetTrackedFeaturesPrev(),
                                      papp.GetCTCstPtr()->GetTrackedFeaturesCurr(),
                                      2,
                                      CV_RGB(0,0,255), CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));
        }

        if (viz_rois) papp.GetRECstPtr()->DrawROIs(debug_frame, true);
        if (frame.data) cv::imshow("Original", frame);
        if (diff_frame.data) cv::imshow("DiffStab", diff_frame);
        if (debug_frame.data) cv::imshow("Debug", debug_frame);
        cv::waitKey(5);
        TICK("ML_Visualization");
      }
      LOG(INFO) << "[ML] Timing info" << std::endl
                << StepBenchmarker::GetInstance().getstr(clear);
      //ticker.dump(clear);
      while (display && pause) cv::waitKey(100);
      frame_counter++;
      LOG(INFO) << "[ML] " << frame_counter << " / " << num_frames;
      if (!use_webcam && loop && frame_counter == num_frames-1)
      {
        frame_counter = start_frame;
        capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
      }
      StepBenchmarker::GetInstance().reset();
    }
  } catch (const std::runtime_error& ex) {
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "[ML] Exception: " << ex.what();
    if (capture.isOpened()) capture.release();
    LOG(INFO) << "[ML] Timing info" << std::endl
              << StepBenchmarker::GetInstance().getstr(clear);
    return 1;
  }
  if (capture.isOpened()) capture.release();

  if (eval_mode && papp.Alive())
  {
    LOG(INFO) << "[ML] Evaluation failed";
    std::ofstream metadata_file(eval_file + std::string(".txt"), std::ios::out);
    // uid: 0 means no target was found in this sequence
    metadata_file << 0 << std::endl;
    metadata_file.close();
  }
  return 0;
}

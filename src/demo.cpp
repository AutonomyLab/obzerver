#include <iostream>

#include "glog/logging.h"
#include "opencv2/features2d/features2d.hpp"

#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"

void mouseCallback(int event, int x, int y, int idonnow, void* data) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    bool* pause = (bool*) data;
    *pause = !(*pause);
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    std::cout << "Right click at " << x << " , " << y << std::endl;
  }
}

void init_gui(cv::VideoCapture& capture, const bool pause) {
  const long int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
  cv::namedWindow("DiffStab", cv::WINDOW_NORMAL | cv::WINDOW_OPENGL);
  cv::namedWindow("Debug", cv::WINDOW_NORMAL | cv::WINDOW_OPENGL);
  cv::createTrackbar("Browse", "Original", 0, num_frames, trackbarCallback, &capture);
  cv::setMouseCallback("Original", mouseCallback, (void*) &pause);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  cv::CommandLineParser cmd(argc, argv,
              "{ v  | video | | specify video file }"
              "{ d  | display | false | Show visualization }"
              "{ c  | clear | false | Clear Terminal }"
              "{ p  | pause | false | Start in pause mode }"
              "{ ds | downsample | 1.0 | downsample factor }"
              "{ l  | logfile | | specify log file (empty: log to stderr)}"
              "{ nf | numfeatures | 300 | Number of features to track for stablization}"
              "{ np | numparticles | 1000 | Number of particles}"
              "{ hi | history | 90 | Length of history (frames) }"
              "{ h | help | false | print help message }"
  );

  /* Params and Command Line */

  const std::string video_src = cmd.get<std::string>("video");
  const bool display = cmd.get<bool>("display");
  const bool clear = cmd.get<bool>("clear");
  const float downsample_factor = cmd.get<unsigned int>("downsample");
  bool pause = cmd.get<bool>("pause");
  const std::string logfile = cmd.get<std::string>("logfile");

  const std::size_t param_max_features = cmd.get<std::size_t>("numfeatures");
  const std::size_t param_num_particles = cmd.get<std::size_t>("numparticles");
  const std::size_t param_hist_len = cmd.get<std::size_t>("history");
  const std::size_t param_pylk_winsize = 30;
  const unsigned int param_pylk_iters = 30;
  const double param_pylk_eps = 0.01;

  const int param_ffd_threshold = 30;

  if (cmd.get<bool>("help") || video_src.empty())
  {
    cmd.printParams();
    return 1;
  }

  /* Logger */

  if (logfile.empty()) {
    google::LogToStderr();
  } else {
    std::cout << "Logging to: " <<  logfile << std::endl;
    google::SetLogDestination(google::GLOG_INFO, logfile.c_str());
  }

  /* Variables */
  cv::Mat frame;
  cv::Mat frame_gray;
  StepBenchmarker& ticker = StepBenchmarker::GetInstance();
  cv::VideoCapture capture;
  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);
  CameraTracker camera_tracker(param_hist_len, feature_detector, param_max_features, param_pylk_winsize, param_pylk_iters, param_pylk_eps);

  LOG(INFO) << "Video Source: " << video_src;

  try {
    if (!capture.open(video_src)) {
      LOG(ERROR) << "Can not open file: " << video_src;
      return 1;
    }
    LOG(INFO) << "Openning file: " << video_src << " frames: " << capture.get(CV_CAP_PROP_FRAME_COUNT);
    if (display) {
      init_gui(capture, pause);
    }
    for (unsigned long int frame_counter = 0; ; frame_counter++) {
      ticker.reset();
      LOG(INFO) << "Frame: " << frame_counter;
      if (!capture.read(frame)) break;
      if (display && (frame_counter % 10 == 0)) cv::setTrackbarPos("Browse", "Original", capture.get(CV_CAP_PROP_POS_FRAMES));
      ticker.tick("Frame Capture");
      cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
      ticker.tick("Frame 2 Gray");
      camera_tracker.Update(frame_gray);
      ticker.dump(clear);
      while (display && pause) cv::waitKey(100);
    }
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "Exception: " << ex.what();
    if (capture.isOpened()) capture.release();
    ticker.dump();
    return 1;
  }
  return 0;
}

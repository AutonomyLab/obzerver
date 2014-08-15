#include <iostream>

#include "glog/logging.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/camera_tracker.hpp"
#include "obzerver/object_tracker.hpp"
#include "obzerver/self_similarity.hpp"
#include "obzerver/fft.hpp"

void mouseCallback(int event, int x, int y, int idonnow, void* data) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    bool* pause = (bool*) data;
    *pause = !(*pause);
  } else if (event == cv::EVENT_RBUTTONDOWN) {
    std::cout << "Right click at " << x << " , " << y << std::endl;
  }
}

void init_gui(cv::VideoCapture& capture, bool* pause) {
;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  cv::CommandLineParser cmd(argc, argv,
              "{ v  | video | | specify video file }"
              "{ d  | display | false | Show visualization }"
              "{ c  | clear | false | Clear Terminal }"
              "{ p  | pause | false | Start in pause mode }"
              "{ ds | downsample | 1.0 | downsample (resize) factor (0.5: half) }"
              "{ l  | logfile | | specify log file (empty: log to stderr)}"
              "{ nf | numfeatures | 300 | Number of features to track for stablization}"
              "{ np | numparticles | 1000 | Number of particles}"
              "{ hi | history | 90 | Length of history (frames) }"
              "{ k  | skip | 0 | Starting frame }"
              "{ f  | fps | 30.0 | frames per second }"
              "{ h  | help | false | print help message }"
  );

  /* Params and Command Line */

  const std::string video_src = cmd.get<std::string>("video");
  const bool display = cmd.get<bool>("display");
  const bool clear = cmd.get<bool>("clear");
  const float downsample_factor = cmd.get<float>("downsample");
  bool pause = cmd.get<bool>("pause");
  const std::string logfile = cmd.get<std::string>("logfile");
  const unsigned long int start_frame = cmd.get<unsigned long int>("skip");
  const float fps = cmd.get<float>("fps");

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
  unsigned long int frame_counter = 0;
  cv::Mat frame;
  cv::Mat frame_gray;
  cv::Mat debug_frame;
  cv::Mat stab_frame;

  StepBenchmarker& ticker = StepBenchmarker::GetInstance();
  cv::VideoCapture capture;
  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);
  CameraTracker camera_tracker(param_hist_len, feature_detector, param_max_features, param_pylk_winsize, param_pylk_iters, param_pylk_eps);
  trackbar_data_t trackbar_data(&capture, &frame_counter);
  ObjectTracker object_tracker(param_num_particles, param_hist_len);
  SelfSimilarity self_similariy(param_hist_len);
  Periodicity periodicity(param_hist_len, fps);

  LOG(INFO) << "Video Source: " << video_src;

  try {
    if (!capture.open(video_src)) {
      LOG(ERROR) << "Can not open file: " << video_src;
      return 1;
    }
    if (start_frame > 0 && start_frame < capture.get(CV_CAP_PROP_FRAME_COUNT)) {
      frame_counter = start_frame;
      capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
    }
    LOG(INFO) << "Openning file: " << video_src << " frames: " << capture.get(CV_CAP_PROP_FRAME_COUNT);
    if (display) {
      const long int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
      cv::namedWindow("Original", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
      cv::namedWindow("DiffStab", cv::WINDOW_NORMAL | cv::WINDOW_OPENGL);
      cv::namedWindow("Debug", cv::WINDOW_NORMAL | cv::WINDOW_OPENGL);
      if (num_frames > 0) {
        cv::createTrackbar("Browse", "Original", 0, num_frames, trackbarCallback, &trackbar_data);
        cv::setTrackbarPos("Browse", "Original", frame_counter);
      }
      cv::setMouseCallback("Original", mouseCallback, (void*) &pause);
    }

    ticker.reset();
    while (capture.read(frame)) {
      ticker.tick("Frame Capture");
      if (downsample_factor < 1.0 && downsample_factor > 0.0) {
        cv::resize(frame, frame, cv::Size(0, 0), downsample_factor, downsample_factor, cv::INTER_CUBIC);
        ticker.tick("Downsampling");
      }
      LOG(INFO) << "Frame: " << frame_counter << " [" << frame.cols << " x " << frame.rows << "]";
      if (display && (frame_counter % 10 == 0)) cv::setTrackbarPos("Browse", "Original", frame_counter);
      cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
      ticker.tick("Frame 2 Gray");
      bool ct_success = camera_tracker.Update(frame_gray);
      cv::Point2d center;
      double _w=0.0, _h=0.0;
      if (!ct_success) {
        LOG(WARNING) << "Camera Tracker Failed";
      } else {
        object_tracker.Update(camera_tracker.GetStablized(),
                              camera_tracker.GetLatestDiff(),
                              camera_tracker.GetLatestSOF(),
                              camera_tracker.GetLatestCameraTransform());
        if (object_tracker.GetStatus() != TRACKING_STATUS_TRACKING) {
          if (!self_similariy.IsEmpty()) self_similariy.Reset();
        } else {
          self_similariy.Update(camera_tracker.GetStablized()(object_tracker.GetBoundingBox()).clone());
          if (self_similariy.IsFull()) {
            periodicity.Update(self_similariy.GetSimMatrix());
            LOG(INFO) << "Dominant Frequency: " << periodicity.GetDominantFrequency();
          }
        }
//        center.x = sampler.Integrate(integrand_mean_x, NULL);
//        center.y = sampler.Integrate(integrand_mean_y, NULL);
//        _w = sqrt(sampler.Integrate(integrand_var_x, (void*) &(center.x)));
//        _h = sqrt(sampler.Integrate(integrand_var_y, (void*) &(center.y)));
//        LOG(INFO) << center << " : " << _w<< " - " <<  _h;
      }

      if (display) {
        cv::Mat diff_frame = camera_tracker.GetLatestDiff();
        cv::Mat debug_frame = camera_tracker.GetStablized();
        object_tracker.DrawParticles(debug_frame);
        if (camera_tracker.GetTrackedFeaturesCurr().size()) {          
          drawFeaturePointsTrajectory(frame,
                                      camera_tracker.GetHomographyOutliers(),
                                      camera_tracker.GetTrackedFeaturesPrev(),
                                      camera_tracker.GetTrackedFeaturesCurr(),
                                      2,
                                      CV_RGB(127,127,127), CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));
        }

        cv::rectangle(diff_frame, cv::Rect(center.x - _w/2, center.y-_h/2, _w, _h), CV_RGB(255, 255, 255));
        //cv::circle(diff_frame, center, 10, CV_RGB(255, 255, 255));
        if (frame.data) cv::imshow("Original", frame);
        if (diff_frame.data) cv::imshow("DiffStab", diff_frame);
        if (debug_frame.data) cv::imshow("Debug", debug_frame);
        cv::waitKey(10);
        ticker.tick("Visualization");
      }
      ticker.dump(clear);
      while (display && pause) cv::waitKey(100);
      frame_counter++;
      ticker.reset();
    }
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "Exception: " << ex.what();
    if (capture.isOpened()) capture.release();
    ticker.dump();
    return 1;
  }
  return 0;
}

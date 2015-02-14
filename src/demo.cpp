#include <iostream>
#include <iomanip>

#include "glog/logging.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

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

int main(int argc, char* argv[]) {

  cv::CommandLineParser cmd(argc, argv,
                            "{ v  | video | | specify video file}"
                            "{ c  | cascade | | specify icf cascade file}"
                            "{ d  | display | false | Show visualization }"
                            "{ cl  | clear | false | Clear Terminal }"
                            "{ p  | pause | false | Start in pause mode }"
                            "{ ds | downsample | 1.0 | downsample (resize) factor (0.5: half) }"
                            "{ l  | logfile | | specify log file (empty: log to stderr)}"
                            "{ nf | numfeatures | 300 | Number of features to track for stablization}"
                            "{ np | numparticles | 1000 | Number of particles}"
                            "{ hi | history | 90 | Length of history (frames) }"
                            "{ k  | skip | 0 | Starting frame }"
                            "{ f  | fps | 30.0 | frames per second }"
                            "{ e  | eval | false | evaluation mode }"
                            "{ efl | decision_f_low | 1.0 | evaluation min frequency }"
                            "{ efh | decision_f_high | 4.0 | evaluation max frequency }"
                            "{ ep  | eval_file | /tmp/eval.jpg | image file to dump the decision bounding box to }"
                            "{ h  | help | false | print help message }"
                            );

  /* Params and Command Line */

  const std::string video_src = cmd.get<std::string>("video");
  const std::string cascade_src = cmd.get<std::string>("cascade");
  const bool display = cmd.get<bool>("display");
  const bool clear = cmd.get<bool>("clear");
  const float downsample_factor = cmd.get<float>("downsample");
  bool pause = cmd.get<bool>("pause");
  const std::string logfile = cmd.get<std::string>("logfile");
  const std::size_t start_frame = cmd.get<std::size_t>("skip");
  const float fps = cmd.get<float>("fps");

  const std::size_t param_max_features = cmd.get<std::size_t>("numfeatures");
  const std::size_t param_num_particles = cmd.get<std::size_t>("numparticles");
  const std::size_t param_hist_len = cmd.get<std::size_t>("history");
  const std::size_t param_pylk_winsize = 30;
  const unsigned int param_pylk_iters = 30;
  const double param_pylk_eps = 0.01;

  const bool eval_mode = cmd.get<bool>("eval");
  const float decision_f_low = cmd.get<float>("decision_f_low");
  const float decision_f_high = cmd.get<float>("decision_f_high");
  const std::string eval_file = cmd.get<std::string>("eval_file");

  const int param_ffd_threshold = 30;

  if (cmd.get<bool>("help") || video_src.empty())
  {
    cmd.printParams();
    return 1;
  }

  const bool use_webam  = (video_src.compare("cam") == 0);

  /* Logger */

  obz::log_config(argv[0], logfile);

  /* Variables */
  unsigned long int frame_counter = 0;
  cv::Mat frame;
  cv::Mat frame_gray;
  cv::Mat debug_frame;
  cv::Mat stab_frame;

  StepBenchmarker& ticker = StepBenchmarker::GetInstance();
  cv::VideoCapture capture;
  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::FastFeatureDetector(param_ffd_threshold, true);

//  cv::Ptr<ccv::ICFCascadeClassifier> ccv_icf_ptr = 0;
  //  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::BRISK(param_ffd_threshold);
  //  cv::Ptr<cv::FeatureDetector> feature_detector = new cv::GoodFeaturesToTrackDetector(param_max_features);
  obz::CameraTracker camera_tracker(param_hist_len, feature_detector, param_max_features, param_pylk_winsize, param_pylk_iters, param_pylk_eps);
  obz::ROIExtraction roi_extraction(0.04, 10, cv::Size(0, 0),
                                    cv::Size(200, 200), 1.0, 1.1, 1.0, 2);
  obz::util::trackbar_data_t trackbar_data(&capture, &frame_counter);



  LOG(INFO) << "Video Source: " << video_src;
  LOG(INFO) << "Feature Detector: " << feature_detector->name();

//  if (!cascade_src.empty())
//  {
//    LOG(INFO) << "Cascade File: " << cascade_src;
//    try
//    {
//      ccv_icf_ptr = new ccv::ICFCascadeClassifier(cascade_src);
//    }
//    catch (const std::exception& e)
//    {
//      LOG(ERROR) << "Initializing cascade classifier failed: " << e.what();
//      ccv_icf_ptr = 0;
//    }
//  }

//  obz::PFObjectTracker object_tracker(param_num_particles, param_hist_len, fps, ccv_icf_ptr);
  obz::MultiObjectTracker multi_object_tracker(90, fps, 60);

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

  try {
    if (use_webam && !capture.open(0)) {
      LOG(ERROR) << "Can not webcam";
      return 1;
    } else if (!use_webam && !capture.open(video_src)) {
      LOG(ERROR) << "Can not open file: " << video_src;
      return 1;
    }
    if (!use_webam && start_frame > 0 && start_frame < capture.get(CV_CAP_PROP_FRAME_COUNT)) {
      frame_counter = start_frame;
      capture.set(CV_CAP_PROP_POS_FRAMES, frame_counter);
    }
    if (!use_webam) {
      LOG(INFO) << "Openning file: " << video_src << " frames: " << capture.get(CV_CAP_PROP_FRAME_COUNT);
    }
    if (display) {
      const long int num_frames = use_webam ? 0 : capture.get(CV_CAP_PROP_FRAME_COUNT);
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
      if (downsample_factor < 1.0 && downsample_factor > 0.0) {
        cv::resize(frame, frame, cv::Size(0, 0), downsample_factor, downsample_factor, cv::INTER_CUBIC);
        ticker.tick("ML_Downsampling");
      }
      LOG(INFO) << "Frame: " << frame_counter << " [" << frame.cols << " x " << frame.rows << "]";
      if (display && !use_webam && (frame_counter % 10 == 0)) cv::setTrackbarPos("Browse", "Original", frame_counter);
      cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
      ticker.tick("ML_Frame_2_Gray");
      bool ct_success = camera_tracker.Update(frame_gray, frame);
      cv::Point2d center;
      double _w=0.0, _h=0.0, _f=-1.0;
      if (!ct_success) {
        LOG(WARNING) << "Camera Tracker Failed";
        // TODO
      } else {
        if (roi_extraction.Update(camera_tracker.GetTrackedFeaturesCurr(),
                                  camera_tracker.GetTrackedFeaturesPrev(),
                                  camera_tracker.GetLatestDiff()))
        {
          obz::rect_vec_t rois;
          roi_extraction.GetValidBBs(rois);
          multi_object_tracker.Update(rois,
                                      frame,
                                      camera_tracker.GetLatestCameraTransform().inv());
        }
//        object_tracker.Update2(camera_tracker.GetStablizedGray(), // TODO
//                               camera_tracker.GetLatestDiff(),
//                               //                              camera_tracker.GetLatestSOF(),
//                               camera_tracker.GetLatestCameraTransform(),
//                               camera_tracker.GetStablizedRGB());


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

        //        if (object_tracker.GetStatus() != TRACKING_STATUS_TRACKING) {
        //          if (!self_similariy.IsEmpty()) self_similariy.Reset();
        //        } else {
        //          self_similariy.Update(camera_tracker.GetStablized()(object_tracker.GetBoundingBox()).clone());
        //          if (self_similariy.IsFull()) {
        //            periodicity.Update(self_similariy.GetSimMatrix());
        //            LOG(INFO) << "Dominant Frequency: " << periodicity.GetDominantFrequency();
        //          }
        //        }
        //        center.x = sampler.Integrate(integrand_mean_x, NULL);
        //        center.y = sampler.Integrate(integrand_mean_y, NULL);
        //        _w = sqrt(sampler.Integrate(integrand_var_x, (void*) &(center.x)));
        //        _h = sqrt(sampler.Integrate(integrand_var_y, (void*) &(center.y)));
        //        LOG(INFO) << center << " : " << _w<< " - " <<  _h;
      }

      if (display) {

        cv::Mat diff_frame = camera_tracker.GetLatestDiff();
        diff_frame.convertTo(diff_frame, CV_8UC1, 5.0);
        //        cv::Mat diff_frame = camera_tracker.GetLatestSOF();
        //        cv::Mat diff_frame;
        //        if (object_tracker.IsTracking()) {
        //          diff_frame = object_tracker.GetObject().GetSelfSimilarity().GetSimMatrixRendered();
        //          cv::imwrite("data/sim.bmp", diff_frame);
        //        }
        cv::Mat debug_frame = camera_tracker.GetStablizedGray();
        //roi_extraction.DrawROIs(frame);
        multi_object_tracker.DrawTracks(frame);
//        object_tracker.DrawParticles(debug_frame);

        //        if (camera_tracker.GetTrackedFeaturesCurr().size()) {
        //          drawFeaturePointsTrajectory(frame,
        //                                      camera_tracker.GetHomographyOutliers(),
        //                                      camera_tracker.GetTrackedFeaturesPrev(),
        //                                      camera_tracker.GetTrackedFeaturesCurr(),
        //                                      2,
        //                                      CV_RGB(0,0,255), CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));
        //        }

        cv::rectangle(diff_frame, cv::Rect(center.x - _w/2, center.y-_h/2, _w, _h), CV_RGB(255, 255, 255));

        std::stringstream ss;
        ss << std::setprecision(5) << "Periodicity: " << _f;
        cv::putText(frame, ss.str(), cv::Point(40,40), 1, CV_FONT_HERSHEY_PLAIN, cv::Scalar(255, 0, 0));
        //cv::circle(diff_frame, center, 10, CV_RGB(255, 255, 255));
        if (frame.data) cv::imshow("Original", frame);
        if (diff_frame.data) cv::imshow("DiffStab", diff_frame);
        if (debug_frame.data) cv::imshow("Debug", debug_frame);
        cv::waitKey(10);
        ticker.tick("ML_Visualization");
      }
      LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
      //ticker.dump(clear);
      while (display && pause) cv::waitKey(100);
      frame_counter++;
      ticker.reset();
    }
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "Exception: " << ex.what();
    if (capture.isOpened()) capture.release();
    LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
    return 1;
  }
  if (capture.isOpened()) capture.release();
  LOG(INFO) << "Timing info" << std::endl << ticker.getstr(clear);
  return 0;
}

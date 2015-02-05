#include <iostream>
#include <glog/logging.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "obzerver/logger.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/ccv_wrapper.hpp"

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
              "{ d  | display | false | Show visualization }"
              "{ c  | cascade |  | cascade file }"
              "{ p  | pause | false | Start in pause mode }"
              "{ ds | downsample | 1.0 | downsample (resize) factor (0.5: half) }"
              "{ l  | logfile | | specify log file (empty: log to stderr)}"
              "{ h  | help | false | print help message }"
  );

  /* Params and Command Line */

  const std::string video_src = cmd.get<std::string>("video");
  const std::string cascade_src = cmd.get<std::string>("cascade");
  const bool display = cmd.get<bool>("display");
  const float downsample_factor = cmd.get<float>("downsample");
  const std::string logfile = cmd.get<std::string>("logfile");
  bool pause = cmd.get<bool>("pause");

  if (cmd.get<bool>("help") || video_src.empty() || cascade_src.empty())
  {
    cmd.printParams();
    return 1;
  }

  /* Logger */

  obz_log_config(argv[0], logfile);

  /* Variables */

  std::size_t frame_counter = 0;
  cv::Mat frame;
  StepBenchmarker& ticker = StepBenchmarker::GetInstance();
  cv::VideoCapture capture;
  trackbar_data_t trackbar_data(&capture, &frame_counter);
  ccv::ICFCascadeClassifier icf(cascade_src);

  LOG(INFO) << "Video Source: " << video_src;
  LOG(INFO) << "Cascace Source: " << cascade_src;

  if (!capture.open(video_src)) {
    LOG(ERROR) << "Can not open file: " << video_src;
    return 1;
  }

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

  cv::Rect roi(0, 200, 200, 200);
  std::int32_t ret_value = 0;
  std::vector<ccv::ICFCascadeClassifier::result_t> icf_result_vec;
  try
  {
    if (display)
    {
      const std::size_t num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
      cv::namedWindow("Video", cv::WINDOW_AUTOSIZE | opengl_flags);
      if (num_frames > 0) {
        cv::createTrackbar("Browse", "Video", 0, num_frames, trackbarCallback, &trackbar_data);
        cv::setTrackbarPos("Browse", "Video", frame_counter);
      }
      cv::setMouseCallback("Video", mouseCallback, (void*) &pause);
    }
    ticker.reset();
    while (capture.read(frame))
    {
      ticker.tick("ML_Frame_Capture");
      if (downsample_factor < 1.0 && downsample_factor > 0.0) {
        cv::resize(frame, frame, cv::Size(0, 0), downsample_factor, downsample_factor, cv::INTER_CUBIC);
        ticker.tick("ML_Downsampling");
      }
      const bool is_grayscale = (frame.channels() == 1);

      if (false == is_grayscale)
      {
        // prepare a copy for
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        ticker.tick("ML_Frame_BGR_RGB");
      }

      icf_result_vec.clear();
      const std::size_t num_objects = icf.Detect(frame, icf_result_vec, roi);
      ticker.tick("ML_ICF");

      LOG(INFO) << "Number of Objects: " << num_objects;

      if (display)
      {
        cv::rectangle(frame, roi, cv::Scalar(0, 0, 0));
        for (std::size_t i = 0; i < icf_result_vec.size(); i++)
        {
          cv::rectangle(frame, icf_result_vec[i].bb, cv::Scalar(0, 0, 0));
        }
        if (frame.data) cv::imshow("Video", frame);
        cv::waitKey(10);
        ticker.tick("ML_Visualization");
      }
      LOG(INFO) << std::endl << ticker.getstr(false);
      while (display && pause) cv::waitKey(100);
      frame_counter++;
      ticker.reset();
    }
  }
  catch (const cv::Exception& ex)
  {
    LOG(ERROR) << "OpenCV Exception: " << ex.what();
    ret_value = 1;
  }
  catch (const std::runtime_error& ex)
  {
    LOG(ERROR) << "Runtime Exception: " << ex.what();
    ret_value = 2;
  }

  if (capture.isOpened()) capture.release();
  LOG(INFO) << std::endl << ticker.getstr(false);
  return ret_value;
}

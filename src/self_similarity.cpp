
#include <iomanip>
#include <stdexcept>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
//#include "opencv2/features2d/features2d.hpp"

#include "glog/logging.h"

#include "obzerver/utility.hpp"
#include "obzerver/self_similarity.hpp"



SelfSimilarity::SelfSimilarity(const std::size_t hist_len, const bool debug_mode):
  hist_len(hist_len),
  debug_mode(debug_mode),
  sequence(hist_len),
  ticker(StepBenchmarker::GetInstance())
{
  ;
}

std::size_t SelfSimilarity::GetHistoryLen() const {
  return sequence.size();
}

void SelfSimilarity::Reset() {
  sequence.clear();
}

bool SelfSimilarity::IsFull() const {
  return (sequence.size() == hist_len);
}

bool SelfSimilarity::IsEmpty() const {
  return (sequence.size() == 0);
}

float SelfSimilarity::CalcFramesSimilarity(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& buff, const
                                            unsigned int index) const {
  //CV_Assert(m1.size() == m2.size());
  //cv::absdiff(m1, m2, buff);
  //cv::multiply(buff, buff, buff);
  cv::Mat img, tmpl;
  bool reversed = false;
  int orig_width, orig_height;
  if (m1.cols >= m2.cols && m1.rows >= m2.rows) {
    //img = m1.clone();
    tmpl = m2.clone();
    cv::copyMakeBorder(m1, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    reversed = false;
    orig_width = m1.cols;
    orig_height = m1.rows;
  } else if (m2.cols >= m1.cols && m2.rows >= m1.rows) {
    //img = m2.clone();
    tmpl = m1.clone();
    cv::copyMakeBorder(m2, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    orig_width = m2.cols;
    orig_height = m2.rows;
    reversed = true;
  } else {
    return 1e12;
    //throw std::runtime_error("Please use fixed ratio for your bounding boxes. This implementation can not handle all overlap situations.");
  }

  cv::matchTemplate(img, tmpl, buff, CV_TM_CCORR_NORMED);
  //cv::matchTemplate(img, tmpl, buff, CV_TM_SQDIFF_NORMED);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);

  cv::Rect tmpl_roi;
  tmpl_roi.x = max_loc.x;
  tmpl_roi.y = max_loc.y;
  tmpl_roi.width = std::min(tmpl.cols, orig_width - (tmpl_roi.tl().x - tmpl.cols/2));
  tmpl_roi.height = std::min(tmpl.rows, orig_height - (tmpl_roi.tl().y - tmpl.rows/2));

  // The sizes of two images should be identical
  // TODO: add why
  // TODO: remove clone?
  cv::Mat img_cropped = img(tmpl_roi).clone();
  cv::Mat tmpl_cropped = tmpl(cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height)).clone();

  cv::rectangle(img, tmpl_roi, cv::Scalar(0,0,0));
  cv::rectangle(tmpl, cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height), cv::Scalar(0,0,0));

  //cv::equalizeHist(tmpl(cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height)).clone(), tmpl);
  //cv::absdiff(img(tmpl_roi).clone(), tmpl(cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height)).clone(), buff);
  cv::absdiff(img_cropped, tmpl_cropped, buff);
  //buff = reversed ? img - tmpl : tmpl - img;

  // Remove noise
  cv::threshold(buff, buff, 10, 0, cv::THRESH_TOZERO);

  if (debug_mode) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_0m1.bmp";
    cv::imwrite(ss.str(), m1);

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_1m2.bmp";
    cv::imwrite(ss.str(), m2);

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_2img.bmp";
    cv::imwrite(ss.str(), img.clone());

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_3tpl.bmp";
    cv::imwrite(ss.str(), tmpl.clone());

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_4buf.bmp";
    cv::imwrite(ss.str(), buff.clone());
  }
  return cv::mean(buff)[0];// / float(tmpl_roi.area());

  //return sqrt(min_val);//cv::sum(buff)[0];// / (float) m1.size().area();
}


void SelfSimilarity::Update(const cv::Mat& m, const bool reset) {
  if (reset) Reset();
  sequence.push_front(m);
  Update();
}

void SelfSimilarity::Update() {
  widths.resize(sequence.size());
  heights.resize(sequence.size());
  std::size_t i = 0;
  for (mseq_t::iterator it = sequence.begin(); it != sequence.end(); it++, i++) {
    widths[i] = sequence.at(i).size().width;
    heights[i] = sequence.at(i).size().height;
  }

  const std::size_t w = quick_median(widths);
  const std::size_t h = quick_median(heights);

  LOG(INFO) << "Median Size: " << w << ", " << h << " @ " << sequence.size();

  //sim_matrix = cv::Mat::zeros(sequence.size(), sequence.size(), CV_32FC1);
  sim_matrix = cv::Mat::zeros(sequence.size(), 1, CV_32F);
  //cv::Mat m1_resized = cv::Mat::zeros(h, w, CV_8UC1);
  //cv::Mat m2_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat buff = cv::Mat::zeros(h, w, CV_8UC1);
  for (std::size_t t1 = 0; t1 < sequence.size(); t1++) {
    //for (std::size_t t2 = 0; t2 < sequence.size(); t2++) {
      //cv::resize(sequence.at(t1), m1_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
      //cv::resize(sequence.at(t2), m2_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
      //const float s = CalcFramesSimilarity(m1_resized, m2_resized, buff);
      //const float s = CalcFramesSimilarity(m1, m2, buff);
      const float s = CalcFramesSimilarity(sequence.at(0), sequence.at(t1), buff, t1);
      sim_matrix.at<float>(t1) = s; // nx1 mat
      //sim_matrix.at<float>(t2, t1) = s;
    //}
  }

  if (debug_mode) {
    LOG(INFO) << sim_matrix.col(0).t();
    WriteToDisk("./data");
  }

//  if (IsFull()) {
//    CalcVecDFT()
//  }
}


const cv::Mat& SelfSimilarity::GetSimMatrix() const {
  return sim_matrix;
}

const cv::Mat SelfSimilarity::GetSimMatrixRendered() {
  double max_val = 0.0, min_val = 0.0;
  cv::minMaxLoc(sim_matrix, &min_val, &max_val);
  if (max_val > 0.0) {
    sim_matrix.convertTo(render, CV_8UC1, 255.0 / max_val);
  } else {
    render = cv::Mat::zeros(sim_matrix.size(), CV_8UC1);
  }
  return render;
}

void SelfSimilarity::WriteToDisk(const std::string& path, const std::string& prefix) const {
  unsigned int frame_counter = 0;
  for (mseq_t::const_reverse_iterator it = sequence.rbegin(); it != sequence.rend(); it++, frame_counter++) {
    std::stringstream ss;
    ss << path << "/" << prefix << std::setfill('0') << std::setw(5) <<  frame_counter << ".bmp";
    cv::imwrite(ss.str(), *it);
  }
}

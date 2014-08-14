
#include <iomanip>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "glog/logging.h"

#include "obzerver/utility.hpp"
#include "obzerver/self_similarity.hpp"



SelfSimilarity::SelfSimilarity(const std::size_t hist_len):
  hist_len(hist_len), sequence(hist_len), ticker(StepBenchmarker::GetInstance())
{;}

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

float SelfSimilarity::CalcFramesSimilarity(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& buff, const unsigned int index) const {
  //CV_Assert(m1.size() == m2.size());
  //cv::absdiff(m1, m2, buff);
  //cv::multiply(buff, buff, buff);
  cv::Mat img, tmpl;
  if (m1.cols >= m2.cols && m1.rows >= m2.rows) {
    img = m1;
    tmpl = m2;

  } else if (m2.cols >= m1.cols && m2.rows >= m1.rows) {
    img = m2;
    tmpl = m1;
  } else {
    return 1e12;
  }
  cv::matchTemplate(img, tmpl, buff, CV_TM_SQDIFF);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);

  cv::Rect tmpl_roi;
  tmpl_roi.x = min_loc.x;
  tmpl_roi.y = min_loc.y;
  tmpl_roi.width = std::min(tmpl.cols, img.cols - tmpl_roi.tl().x);
  tmpl_roi.height = std::min(tmpl.rows, img.rows - tmpl_roi.tl().y);
  cv::absdiff(img(tmpl_roi).clone(), tmpl, buff);

  std::stringstream ss;

  ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_roi.bmp";
  cv::imwrite(ss.str(), img(tmpl_roi));

  ss.str("");
  ss << std::setw(5) << std::setfill('0') << "/tmp/"<< index << "_img.bmp";
  cv::imwrite(ss.str(), img);

  ss.str("");
  ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_tpl.bmp";
  cv::imwrite(ss.str(), tmpl);

  return cv::sum(buff)[0];// / float(tmpl_roi.area());
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

  sim_matrix = cv::Mat::zeros(sequence.size(), sequence.size(), CV_32FC1);
  //cv::Mat m1_resized = cv::Mat::zeros(h, w, CV_8UC1);
  //cv::Mat m2_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat buff = cv::Mat::zeros(h, w, CV_8UC1);
  for (std::size_t t1 = 0; t1 < 1/*sequence.size()*/; t1++) {
    for (std::size_t t2 = 0; t2 < sequence.size(); t2++) {
      //cv::resize(sequence.at(t1), m1_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
      //cv::resize(sequence.at(t2), m2_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
      //const float s = CalcFramesSimilarity(m1_resized, m2_resized, buff);
      //const float s = CalcFramesSimilarity(m1, m2, buff);
      const float s = CalcFramesSimilarity(sequence.at(t1), sequence.at(t2), buff, t2);
      sim_matrix.at<float>(t1, t2) = s;
      //sim_matrix.at<float>(t2, t1) = s;
    }
  }
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
    ss << path << prefix << std::setfill('0') << std::setw(5) <<  frame_counter << ".bmp";
    cv::imwrite(ss.str(), *it);
  }
}

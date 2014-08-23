#include <iomanip>
#include <stdexcept>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"


#include "glog/logging.h"

#include "obzerver/utility.hpp"
#include "obzerver/self_similarity.hpp"
#include "obzerver/opencv3-backport/shift.hpp"

class ParallelFrameSimilarity: public cv::ParallelLoopBody {
private:
  CircularBuffer<cv::Mat>& seq;
  cv::Mat& sim;
  cv::Mat m1;
  std::size_t sz;
  cv::Size img_sz;

public:
  ParallelFrameSimilarity(
      CircularBuffer<cv::Mat>& _seq,
      cv::Mat& _sim,
      const cv::Mat& _ff,
      const cv::Size& _img_sz)
    : seq(_seq),
      sim(_sim),
      m1(_ff),
      sz(_seq.size()),
      img_sz(_img_sz)
  {
  }

  virtual void operator ()(const cv::Range& range) const
  {
    cv::Mat buff;
    cv::Mat m2;
    for (int i = range.start; i < range.end; i++) {
      cv::resize(seq.at(i), m2, img_sz, 0.0, 0.0, cv::INTER_CUBIC);
      const float s = SelfSimilarity::CalcFramesSimilarity(m1, m2, buff, (unsigned int) i, false);
      sim.at<float>(0, i) = s;
      sim.at<float>(i, 0) = s;
    }
  }
};

SelfSimilarity::SelfSimilarity(const std::size_t hist_len, const bool debug_mode):
  hist_len(hist_len),
  debug_mode(debug_mode),
  sequence(hist_len),
  sim_matrix(cv::Mat::zeros(hist_len, hist_len, CV_32F)),
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

float SelfSimilarity::CalcFramesSimilarity(const cv::Mat& m1,
                                           const cv::Mat& m2,
                                           cv::Mat& buff,
                                           const unsigned int index,
                                           bool debug_mode = false)
{
  cv::Mat img, tmpl;
  int orig_width, orig_height;
  if ( (m1.cols * m1.rows) >= (m2.cols * m2.rows)) {
    m2.copyTo(tmpl);
    cv::copyMakeBorder(m1, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    orig_width = m1.cols;
    orig_height = m1.rows;
  } else {
    m1.copyTo(tmpl);
    cv::copyMakeBorder(m2, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    orig_width = m2.cols;
    orig_height = m2.rows;
  }
  cv::matchTemplate(img, tmpl, buff, CV_TM_SQDIFF);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);

  if (debug_mode) {
    // img coordinate system is the global coordinate system here
    // from 0,0 -> img.w + tmpl.w, img.h + tmpl.h

    // overlap in global coordinate frame
    // & is intersection of two rects
    cv::Rect tmpl_overlap =
        cv::Rect(tmpl.cols/2, tmpl.rows/2, orig_width, orig_height) & // original image in global coordinates
        cv::Rect(min_loc, cv::Size(tmpl.cols, tmpl.rows)); // matched tmpl in global coordinates

    cv::Rect tmpl_roi(
          tmpl_overlap.x - min_loc.x,
          tmpl_overlap.y - min_loc.y,
          tmpl_overlap.width,
          tmpl_overlap.height
          );

    cv::Mat img_cropped = img(tmpl_overlap).clone(); // This is ok (both in global coordinates)

    cv::Mat tmpl_cropped = tmpl(tmpl_roi).clone();

    cv::rectangle(img, tmpl_overlap, cv::Scalar(0,0,0));
    cv::rectangle(tmpl, tmpl_roi, cv::Scalar(0,0,0));
    CV_Assert(img_cropped.size() == tmpl_cropped.size());
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

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_5m2_cropped.bmp";
    cv::imwrite(ss.str(), m2);

  }

  return min_val;
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

  const std::size_t w = quick_median(widths)* 0.5;
  const std::size_t h = quick_median(heights) * 0.5;

//  LOG(INFO) << "Median Size: " << w << ", " << h << " @ " << sequence.size();

  cv::Mat m1_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv3::shift(sim_matrix, sim_matrix, cv::Point2f(1.0,1.0));

#if 1
  cv::resize(sequence.at(0), m1_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
  cv::parallel_for_(
        cv::Range(0, sequence.size() - 1),
        ParallelFrameSimilarity(
          sequence,
          sim_matrix,
          m1_resized,
          cv::Size(w,h)
          ), 4 // Use Four Threads
        );
#else
  cv::Mat m2_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat buff = cv::Mat::zeros(h, w, CV_8UC1);
  cv::resize(sequence.at(0), m1_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);

  for (std::size_t t1 = 0; t1 < sequence.size(); t1++) {
    cv::resize(sequence.at(t1), m2_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
    //const float s = CalcFramesSimilarity(sequence.at(0), sequence.at(t1), buff, t1);
    const float s = CalcFramesSimilarity(m1_resized, m2_resized, buff, t1);
    sim_matrix.at<float>(0, t1) = s;
    sim_matrix.at<float>(t1, 0) = s;
  }
#endif

  if (debug_mode) {
    LOG(INFO) << sim_matrix;
    WriteToDisk("./data");
  }

  ticker.tick("  [SS] Self Similarity Update");
}

const cv::Mat& SelfSimilarity::GetSimMatrix() const {
  return sim_matrix;
}

cv::Mat SelfSimilarity::GetSimMatrixRendered() const {
  cv::Mat render;
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

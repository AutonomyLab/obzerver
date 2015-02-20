#include <iomanip>
#include <stdexcept>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#include "glog/logging.h"

#include "obzerver/utility.hpp"
#include "obzerver/self_similarity.hpp"
#include "obzerver/opencv3-backport/shift.hpp"

namespace obz
{

class ParallelFrameSimilarity: public cv::ParallelLoopBody {
private:
  std::size_t t1;
  const obz::mseq_t& seq;
  cv::Mat& sim;
  cv::Mat m1;
  std::size_t sz;
  cv::Size img_sz;

public:
  ParallelFrameSimilarity(
      std::size_t& _t1,
      const obz::mseq_t& _seq,
      cv::Mat& _sim,
      const cv::Mat& _ff,
      const cv::Size& _img_sz) :
    t1(_t1),
    seq(_seq),
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

    float s;
    std::size_t count = 0;
    for (int i = range.start; i < range.end; i+=1) {
      // Used for non-resizable BBs
      // Max-non-SelfSimilarity when using TM_SQ_DIFF
      s = 65025;
      if (seq[i].size().width && seq[i].size().height)
      {
        count++;
        cv::resize(seq[i], m2, img_sz, 0.0, 0.0, cv::INTER_LINEAR);
        s = SelfSimilarity::CalcFramesSimilarity(m1, m2, buff, (unsigned int) i, false);
      }
      sim.at<float>(t1, i) = s;
      sim.at<float>(i, t1) = s;
    }
  }
};

SelfSimilarity::SelfSimilarity(const std::size_t hist_len,
                               const std::uint64_t current_time,
                               const bool debug_mode):
  debug_mode(debug_mode),
  sim_matrix(cv::Mat::zeros(hist_len, hist_len, CV_32F)),
  last_update_time(current_time),
  ticker(StepBenchmarker::GetInstance())
{
  ;
}

float SelfSimilarity::CalcFramesSimilarity(const cv::Mat& m1,
                                           const cv::Mat& m2,
                                           cv::Mat& buff,
                                           const unsigned int index,
                                           bool debug_mode = false)
{
#if 0
  CV_Assert(m1.size() == m2.size());
  cv::matchTemplate(m1, m2, buff, CV_TM_CCOEFF_NORMED);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);
#else
  cv::Mat img, tmpl;
  int orig_width, orig_height;
  if ( (m1.cols * m1.rows) >= (m2.cols * m2.rows)) {
    //m2.copyTo(tmpl);
    tmpl = m2;
    cv::copyMakeBorder(m1, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);

    // TODO: Parameterize this
//    cv::copyMakeBorder(m1, img, 10, 10, 10, 10, cv::BORDER_CONSTANT);
    orig_width = m1.cols;
    orig_height = m1.rows;
  } else {
    //m1.copyTo(tmpl);
    tmpl = m1;
//    cv::copyMakeBorder(m2, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);

    cv::copyMakeBorder(m2, img, 10, 10, 10, 10, cv::BORDER_WRAP);
    orig_width = m2.cols;
    orig_height = m2.rows;
  }
  cv::matchTemplate(img, tmpl, buff, CV_TM_CCOEFF);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);

//  if (debug_mode) {
    // img coordinate system is the global coordinate system here
    // from 0,0 -> img.w + tmpl.w, img.h + tmpl.h

    // overlap in global coordinate frame
    // & is intersection of two rects
    cv::Rect tmpl_overlap =
        cv::Rect(tmpl.cols/2, tmpl.rows/2, orig_width, orig_height) & // original image in global coordinates
        cv::Rect(max_loc, cv::Size(tmpl.cols, tmpl.rows)); // matched tmpl in global coordinates

    cv::Rect tmpl_roi(
          tmpl_overlap.x - max_loc.x,
          tmpl_overlap.y - max_loc.y,
          tmpl_overlap.width,
          tmpl_overlap.height
          );

    cv::Mat img_cropped = img(tmpl_overlap).clone(); // This is ok (both in global coordinates)

    cv::Mat tmpl_cropped = tmpl(tmpl_roi).clone();

//    cv::rectangle(img, tmpl_overlap, cv::Scalar(0,0,0));
//    cv::rectangle(tmpl, tmpl_roi, cv::Scalar(0,0,0));
    CV_Assert(img_cropped.size() == tmpl_cropped.size());
//    std::stringstream ss;
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_0m1.bmp";
//    cv::imwrite(ss.str(), m1);

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_1m2.bmp";
//    cv::imwrite(ss.str(), m2);

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_2img.bmp";
//    cv::imwrite(ss.str(), img.clone());

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_3tpl.bmp";
//    cv::imwrite(ss.str(), tmpl.clone());

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_4buf.bmp";
//    cv::imwrite(ss.str(), buff.clone());

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_5m2_cropped.bmp";
//    cv::imwrite(ss.str(), m2);

//  }
#endif
  cv::matchTemplate(img, tmpl, buff, CV_TM_CCOEFF);
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);
  return max_val;// / static_cast<float>(m1.rows * m1.cols);
}


void SelfSimilarity::Update(obz::mseq_t& sequence, std::uint64_t current_time, const std::string& debug_prefix) {
  LOG(INFO) << "[SS] Size: " << sequence.size();
  widths.resize(sequence.size());
  heights.resize(sequence.size());
  std::size_t i = 0;
  for (mseq_t::const_iterator it = sequence.begin(); it != sequence.end(); it++, i++)
  {
    widths[i] = it->size().width;
    heights[i] = it->size().height;
  }

  std::size_t med_w = util::quick_median(widths);
  std::size_t med_h = util::quick_median(heights);
  const std::size_t lag = std::min(current_time - last_update_time, sequence.size());

  LOG(INFO) << "[SS] Median Size: " << med_w << ", " << med_h
            << " @ " << sequence.size() << " lag: " << lag;

  // TODO: PUSH NULL SS
  if (med_w == 0 || med_h == 0)
  {
    LOG(WARNING) << "[SS] Median Size is Zero: " << med_w << " x " << med_h;
    return;
  }

  const std::size_t max_wh = std::max(med_w, med_h);
  if (max_wh > 50)
  {
    med_w *= (50.0 / max_wh);
    med_h *= (50.0 / max_wh);
  }

  for (std::size_t s = 0; s < sequence.size(); s++)
  {
    if ((!sequence[s].rows) || (!sequence[s].cols))
    {
      LOG(WARNING) << "[SS] Area of one BB in sequence is 0";
      return;
    }
    cv::resize(sequence[s], sequence[s], cv::Size2d(med_w, med_h), 0, 0, cv::INTER_LINEAR);
  }

  // We only perform `lag` number of SS calculations
  cv3::shift(sim_matrix, sim_matrix, cv::Point2f(lag, lag));
  cv::Mat buff = cv::Mat::zeros(med_h, med_w, CV_32FC1);

  for (std::size_t t1 = 0; t1 < lag; t1++)
  {
//    cv::Mat m1_resized = cv::Mat::zeros(med_h, med_w, CV_8UC1);
//    cv3::shift(sim_matrix, sim_matrix, cv::Point2f(1.0,1.0));

  #if 0
    if (!sequence[t1].size().area()) return;
    cv::resize(sequence[t1], m1_resized, cv::Size2d(med_w, med_h), 0, 0, cv::INTER_LINEAR);
    cv::parallel_for_(
          cv::Range(0, sequence.size() - 1),
          ParallelFrameSimilarity(
            t1,
            sequence,
            sim_matrix,
            m1_resized,
            cv::Size(med_w,med_h)
            ), 2 // Use Four Threads
          );
  #else
//    cv::Mat m2_resized = cv::Mat::zeros(med_h, med_w, CV_8UC1);

//    cv::resize(sequence.at(t1), m1_resized, cv::Size2d(med_w, med_h), 0, 0, CV_INTER_LINEAR);

    for (std::size_t t2 = t1; t2 < sequence.size(); t2++) {
//      cv::resize(sequence.at(t2), m2_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
      //const float s = CalcFramesSimilarity(sequence.at(0), sequence.at(t1), buff, t1);
      const float s = CalcFramesSimilarity(sequence[t1], sequence[t2], buff, t1);
      sim_matrix.at<float>(t1, t2) = s;
      sim_matrix.at<float>(t2, t1) = s;
    }
  #endif
  }

  last_update_time = current_time;
  if (/*debug_mode*/ !debug_prefix.empty()) {
    //LOG(INFO) << sim_matrix;
    WriteToDisk(sequence, "./data", debug_prefix);
  }
//  ticker.tick("SS_Self_Similarity_Update");
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

void SelfSimilarity::WriteToDisk(const mseq_t &sequence,
                                 const std::string& path,
                                 const std::string& prefix) const
{
  unsigned int frame_counter = 0;
  for (mseq_t::const_reverse_iterator it = sequence.rbegin(); it != sequence.rend(); it++, frame_counter++) {
    std::stringstream ss;
    ss << path << "/" << prefix << std::setfill('0') << std::setw(5) <<  frame_counter << ".bmp";
    cv::imwrite(ss.str(), *it);
  }
}

}  // namespace obz

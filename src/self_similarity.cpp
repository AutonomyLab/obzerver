#include <iomanip>
#include <stdexcept>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv3-backport/shift.hpp"

#include "glog/logging.h"

#include "obzerver/utility.hpp"
#include "obzerver/self_similarity.hpp"
SelfSimilarity::SelfSimilarity(const std::size_t hist_len, const bool debug_mode):
  hist_len(hist_len),
  debug_mode(debug_mode),
  sequence(hist_len),
  ticker(StepBenchmarker::GetInstance())
{
  sim_matrix = cv::Mat::zeros(hist_len, hist_len, CV_32F);
  background_substractor = new cv::BackgroundSubtractorMOG(hist_len, 5, 0.5);
  feature_detector = new cv::FastFeatureDetector(30, true);
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
  if ( (m1.cols * m1.rows) >= (m2.cols * m2.rows)) {
    //img = m1.clone();
    tmpl = m2.clone();
    cv::copyMakeBorder(m1, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    reversed = false;
    orig_width = m1.cols;
    orig_height = m1.rows;
  } else {//if (m2.cols >= m1.cols && m2.rows >= m1.rows) {
    //img = m2.clone();
    tmpl = m1.clone();
    cv::copyMakeBorder(m2, img, tmpl.rows/2, tmpl.rows/2, tmpl.cols/2, tmpl.cols/2, cv::BORDER_WRAP);
    orig_width = m2.cols;
    orig_height = m2.rows;
    reversed = true;
  } /*else {
    return 1e12;
    //throw std::runtime_error("Please use fixed ratio for your bounding boxes. This implementation can not handle all overlap situations.");
  }*/

  //cv::matchTemplate(img, tmpl, buff, CV_TM_CCORR_NORMED);
  cv::matchTemplate(img, tmpl, buff, CV_TM_SQDIFF);
  //cv::matchTemplate(img, tmpl, buff, CV_TM_SQDIFF_NORMED);
  double max_val = 0.0, min_val = 0.0;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(buff, &min_val, &max_val, &min_loc, &max_loc);

  return min_val;
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

//  LOG(INFO) << "Orig Size: " << orig_width << " " << orig_height;
//  LOG(INFO) << "Big image Size: " << img.size();
//  LOG(INFO) << "Template Size: " << tmpl.size();
//  LOG(INFO) << "Max Loc: " << max_loc;
//  LOG(INFO) << "Overlap Size: " << tmpl_overlap;
//  LOG(INFO) << "TMPL ROI size: " << tmpl_roi;
  // The sizes of two images should be identical
  // TODO: add why
  // TODO: remove clone?
  cv::Mat img_cropped = img(tmpl_overlap).clone(); // This is ok (both in global coordinates)

  cv::Mat tmpl_cropped = tmpl(tmpl_roi).clone();

  cv::rectangle(img, tmpl_overlap, cv::Scalar(0,0,0));
  cv::rectangle(tmpl, tmpl_roi, cv::Scalar(0,0,0));

//  std::vector<cv::KeyPoint> k1;
//  std::vector<cv::Point2f> v1, v1_nxt;
//  std::vector<int> v1_status;
//  feature_detector->detect(img_cropped, k1);
//  //feature_detector->detect(tmpl_cropped, k2);
//  cv::KeyPoint::convert(k1, v1);
//  //cv::KeyPoint::convert(k2, v2);
//  cv::calcOpticalFlowPyrLK(img, tmpl, v1, v1_nxt, v1_status, cv::noArray(), cv::Size(5,5), 3,
//                           cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 1000, 0.01));


//  drawFeaturePointsTrajectory(img, v1, v1_nxt, 2, cv::Scalar(0,0,0), cv::Scalar(0,0,0), cv::Scalar(0,0,0));
  //cv::equalizeHist(tmpl(cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height)).clone(), tmpl);
  //cv::absdiff(img(tmpl_roi).clone(), tmpl(cv::Rect(0,0,tmpl_roi.width, tmpl_roi.height)).clone(), buff);
  CV_Assert(img_cropped.size() == tmpl_cropped.size());

  //cv::absdiff(img_cropped, tmpl_cropped, buff);
  //cv::matchTemplate(img_cropped, tmpl_cropped, buff, CV_TM_CCORR);
  //buff = reversed ? img_cropped - tmpl_cropped : tmpl_cropped - img_cropped;

  cv::absdiff(img_cropped, tmpl_cropped, buff);
  //cv::Scalar _m, _s;
  //cv::meanStdDev(buff, _m, _s);
  // Remove noise  
  //cv::threshold(buff, buff, _m[0] + _s[0], 0, cv::THRESH_TOZERO);

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

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_5m2_cropped.bmp";
    cv::imwrite(ss.str(), m2);
  }

  //return cv::norm(img_cropped, tmpl_cropped, cv::NORM_HAMMING2);
  //return sqrt(min_val);

  return cv::sum(buff)[0];// / float(tmpl_roi.area());

  //return sqrt(min_val);//cv::sum(buff)[0];// / (float) m1.size().area();
}

float SelfSimilarity::CalcFramesSimilarity2(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& buff, const
                                            unsigned int index)
{
  CV_Assert(m1.size() == m2.size());
  cv::Mat _m1, _m2;

//  cv::Mat fg_mask;
//  background_substractor->operator ()(m2, fg_mask);

//  cv::Mat mask;
//  background_substractor->operator ()(m2, mask);
//  cv::Mat affine = cv::estimateRigidTransform(m1, m2, true);

  //cv::matchTemplate(m1.clone(), m2.clone(), buff, CV_TM_CCOEFF_NORMED);
  //cv::Mat buff2;
//  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30,30));

//  cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, element);
  buff.convertTo(buff, CV_16S);
  cv::subtract(m2, m1, buff);
  buff = cv::abs(buff);
 // dumpCvMatInfo(buff2);

  if (debug_mode) {
//    cv::rectangle(_m1, r1, cv::Scalar(0, 0, 0));
//    cv::rectangle(_m2, r2, cv::Scalar(0, 0, 0));
    std::stringstream ss;
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_0bg.bmp";
//    cv::imwrite(ss.str(), mask);
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_0m1.bmp";
    cv::imwrite(ss.str(), m1.clone());

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_1m2.bmp";
    cv::imwrite(ss.str(), m2.clone());

    ss.str("");
    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_2buf.bmp";
    cv::imwrite(ss.str(), buff);

//    ss.str("");
//    ss << std::setw(5) << std::setfill('0') << "/tmp/" << index << "_3mask.bmp";
//    cv::imwrite(ss.str(), fg_mask);
  }
//  LOG(INFO) << "Affine: " << affine;
  return cv::sum(buff)[0]; // TODO: FIXME
}

float SelfSimilarity::CalcFramesSimilarity3(const cv::Mat& m1, const cv::Mat& m2, cv::Mat& buff, const
                                            unsigned int index) const
{
//  cv::Mat flow, flowMag;
//  cv::calcOpticalFlowFarneback(m1, m2, flow, 0.5, 3, 3, 3, 9, 1.9, 0);
//  std::vector<cv::Mat> flowChannels;
//  split(flow, flowChannels);
//  cv::magnitude(flowChannels[0], flowChannels[1], flowMag);

  return 0.0;
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

  const std::size_t w = quick_median(widths) * 0.5;
  const std::size_t h = quick_median(heights) * 0.5;

  LOG(INFO) << "Median Size: " << w << ", " << h << " @ " << sequence.size();

  //cv::Ptr<cv::BackgroundSubtractor> background_substractor = new cv::BackgroundSubtractorMOG2(hist_len, 10);

  cv::Mat m1_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat m2_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat buff = cv::Mat::zeros(h, w, CV_8UC1);

  cv3::shift(sim_matrix, sim_matrix, cv::Point2f(1.0,1.0));

  cv::resize(sequence.at(0), m1_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
  for (std::size_t t1 = 0; t1 < sequence.size(); t1++) {
    cv::resize(sequence.at(t1), m2_resized, cv::Size2d(w, h), 0, 0, CV_INTER_CUBIC);
    //const float s = CalcFramesSimilarity(sequence.at(0), sequence.at(t1), buff, t1);
    const float s = CalcFramesSimilarity(m1_resized, m2_resized, buff, t1);
    sim_matrix.at<float>(0, t1) = s;
    sim_matrix.at<float>(t1, 0) = s;
  }

//  LOG(INFO) << sim_matrix;
//  std::vector<cv::Point2f> corners;
//  cv::goodFeaturesToTrack(m1_resized, corners, 10, 0.05, 2);
//  drawFeaturePoints(m1_resized, corners, cv::Scalar(0,0), 2);
//  LOG(INFO) << "Corners: " << corners.size();
//  for (std::size_t t1 = 0; t1 < sequence.size(); t1++) {
    //for (std::size_t t2 = 0; t2 < sequence.size(); t2++) {

      //const float s = CalcFramesSimilarity(m1_resized, m2_resized, buff);
      //const float s = CalcFramesSimilarity(m1, m2, buff);

//      const float s = CalcFramesSimilarity2(m1_resized, m2_resized, buff, t1);

      //sim_matrix.at<float>(t2, t1) = s;
    //}
//  }

  if (debug_mode) {
//    cv::imwrite("/tmp/00000000.bmp", fg_mask);
//    background_substractor->getBackgroundImage(bg_img);
//    cv::imwrite("/tmp/00000001.bmp", bg_img);
    LOG(INFO) << sim_matrix.col(0).t();
    WriteToDisk("./data");
  }

//  if (IsFull()) {
//    CalcVecDFT()
//  }
}

void SelfSimilarity::Update2() {
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

  cv::Mat m1_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat m2_resized = cv::Mat::zeros(h, w, CV_8UC1);
  cv::Mat buff = cv::Mat::zeros(h, w, CV_8UC1);
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

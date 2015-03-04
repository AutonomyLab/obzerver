#include "obzerver/utility.hpp"
#include "obzerver/roi_extraction.hpp"
#include "obzerver/benchmarker.hpp"

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>

#include <sstream>

namespace obz
{

ROIExtraction::ROIExtraction()
  : dbs_eps_(0.01),
    dbs_min_elements_(100),
    min_roi_sz_(cv::Size(0, 0)),
    max_roi_sz_(cv::Size(100, 100)),
    min_motion_per_pixel_(5.0),
    min_motion_per_feature_(5.0),
    min_optflow_per_feature_(2.0),
    inf_factor_width_(0.0),
    inf_factor_height_(0.0),
    dbs_num_threads_(1.0),
    dbs_engine_(dbs_eps_, dbs_min_elements_, dbs_num_threads_)
{}

ROIExtraction::ROIExtraction(const double dbs_eps,
                             const std::size_t dbs_min_elements,
                             const cv::Size &min_roi_sz,
                             const cv::Size &max_roi_sz,
                             const float min_motion_per_pixel,
                             const float min_motion_per_feature,
                             const float min_optflow_per_feature,
                             const float inflation_width,
                             const float inflation_height,
                             const std::size_t dbs_num_threads)
  : dbs_eps_(dbs_eps),
    dbs_min_elements_(dbs_min_elements),
    min_roi_sz_(min_roi_sz),
    max_roi_sz_(max_roi_sz),
    min_motion_per_pixel_(min_motion_per_pixel),
    min_motion_per_feature_(min_motion_per_feature),
    min_optflow_per_feature_(min_optflow_per_feature),
    inf_factor_width_(inflation_width),
    inf_factor_height_(inflation_height),
    dbs_num_threads_(dbs_num_threads),
    dbs_engine_(dbs_eps_, dbs_min_elements_, dbs_num_threads_)
{
  CV_Assert(inf_factor_width_ >= 0.0 && inf_factor_height_ >= 0.0);
  LOG(INFO) << "[ROI] dps_eps: " << dbs_eps_ << " dbs_min_elements: " << dbs_min_elements_;
}

ROIExtraction::~ROIExtraction()
{}

bool ROIExtraction::Update(
    const obz::pts_vec_t& curr_tracked_features,
    const obz::pts_vec_t& prev_tracked_features,
    const cv::Mat& diff_frame)
{
  CV_Assert(curr_tracked_features.size() == prev_tracked_features.size());
  CV_Assert(diff_frame.channels() == 1);

  dbs_data_.resize(curr_tracked_features.size(), 2);
  for (std::size_t i = 0; i < curr_tracked_features.size(); i++)
  {
    const cv::Point2f& p = curr_tracked_features[i];
    dbs_data_(i, 0) = -1.0 + ((2.0 * p.x) / diff_frame.cols);
    dbs_data_(i, 1) = -1.0 + ((2.0 * p.y) / diff_frame.rows);
  }

  LOG(INFO) << "[ROI] Number of points to cluser: " << curr_tracked_features.size();

  TICK("RO_Features_to_Mat");
  dbs_engine_.init(dbs_eps_, dbs_min_elements_, dbs_num_threads_);
  dbs_engine_.reset();
  dbs_engine_.fit(dbs_data_);
  TICK("RO_Clustering");

  // Skip memory re-allocation as much as possible
//  for (auto &roi_pair: rois_map)
//  {
//    roi_pair.second.reset();
//  }

  rois_map_.clear();

  const clustering::DBSCAN::Labels& cluster_labels = dbs_engine_.get_labels();
  CV_Assert(cluster_labels.size() == curr_tracked_features.size());

  std::size_t num_clusters = 0;
  for (std::size_t i = 0; i < cluster_labels.size(); i++)
  {
    const std::int32_t cid = cluster_labels[i];
    if (cid == -1) continue;
    const cv::Point2f& pc = curr_tracked_features[i];
    const cv::Point2f& pp = prev_tracked_features[i];

    if (diff_frame.at<uchar>(pc) < min_motion_per_feature_)
    {
      continue;
    }

    if (rois_map_.count(cid) == 0)
    {
      rois_map_.insert(obz::roi_pair_t(cid, obz::roi_t()));
      num_clusters++;
    }

    rois_map_[cid].prev_pts.push_back(pp);
    rois_map_[cid].curr_pts.push_back(pc);
  }

  LOG(INFO) << "[ROI] Number of clusters (pre-filter): " << num_clusters;

  num_clusters = 0;
  for (auto &roi_pair: rois_map_)
  {
    obz::roi_t& roi = roi_pair.second;
    if (!roi.curr_pts.size()) continue;

    cv::Rect r = obz::util::ClampRect(
          cv::boundingRect(roi.curr_pts),
          cv::Rect(0, 0, diff_frame.cols-1, diff_frame.rows-1));


    roi.avg_diff_motion_per_pixel = cv::mean(diff_frame(r))[0];

    CV_Assert(roi.curr_pts.size() == roi.prev_pts.size());

    roi.avg_optflow_per_feature = 0.0;
    for (std::size_t i = 0; i < roi.curr_pts.size(); i++)
    {
      const cv::Point2f f = roi.prev_pts[i] - roi.curr_pts[i];
      roi.avg_optflow_per_feature +=  sqrt((f.x * f.x) + (f.y * f.y));
    }
    roi.avg_optflow_per_feature = roi.avg_optflow_per_feature /
        static_cast<float>(roi.curr_pts.size());


    LOG(INFO) << "[ROI] "
              << roi.avg_diff_motion_per_pixel
              << " "
              << roi.avg_optflow_per_feature
              << " "
              << r;

    roi.bb = r;

    if (
        (r.width < min_roi_sz_.width || r.height < min_roi_sz_.height) ||
        (r.width > max_roi_sz_.width || r.height > max_roi_sz_.height) ||
        (roi.avg_diff_motion_per_pixel < min_motion_per_pixel_) /*||
        (roi.optflow_per_feature < min_optflow_per_feature_)*/)
    {
      continue;
    }

    roi.bb = obz::util::ClampRect(
          obz::util::InflateRect(r, inf_factor_width_, inf_factor_height_),
          diff_frame.cols, diff_frame.rows);

    roi.valid = true;
    num_clusters++;
  }
  LOG(INFO) << "[ROI] Number of clusters (post-filter): " << num_clusters;

  TICK("RO_Clustering_PP");
  return num_clusters > 0;
}


void ROIExtraction::DrawROIs(cv::Mat &frame, const bool verbose) const
{
  for (auto& roi_pair: rois_map_)
  {
    std::stringstream text;

    const std::int32_t& cid = roi_pair.first;
    const obz::pts_vec_t& curr_pts = roi_pair.second.curr_pts;
//    const obz::pts_vec_t& prev_pts = roi_pair.second.prev_pts;
    const cv::Rect& bb = roi_pair.second.bb;

    // sorry!
    const cv::Scalar cluster_color = roi_pair.second.valid ?
          (verbose ?
             cv::Scalar(255 * ((cid % 8) & 1), 255 * ((cid % 8) & 2), 255 * ((cid % 8) & 4)) : cv::Scalar(0, 0, 0))
        :
          cv::Scalar(0, 0, 0);

    text << "# " << cid;
//         << " MPP "  << roi_pair.second.avg_diff_motion_per_pixel
//         << " OFPF " << roi_pair.second.avg_optflow_per_feature;
//            << " " << bb;
    if (true || verbose)
    {
      cv::putText(frame, text.str(), cv::Point(bb.x, bb.y - 10), CV_FONT_HERSHEY_PLAIN, 1, cluster_color);
    }
    cv::rectangle(frame, bb, cluster_color, 2);
    for (auto &p: curr_pts)
    {
      cv::circle(frame, p, 3, cv::Scalar(0, 0, 100), -1);
    }
  }
}

std::size_t ROIExtraction::GetValidBBs(rect_vec_t &bb_vec, std::vector<float>& avg_flow) const
{
  std::size_t count = 0;
  for (auto& roi_pair: rois_map_)
  {
    if (!roi_pair.second.valid) continue;
    bb_vec.push_back(roi_pair.second.bb);
    avg_flow.push_back(roi_pair.second.avg_optflow_per_feature);
    count++;
  }
  return count;
}

}  // namespace obz

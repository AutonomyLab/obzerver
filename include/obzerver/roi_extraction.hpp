#ifndef ROI_EXTRACTION
#define ROI_EXTRACTIOn

#include "obzerver/common_types.hpp"
#include "dbscan/dbscan.h"

#include <vector>

namespace obz
{

class ROIExtraction
{
private:
  double dbs_eps_;
  std::size_t dbs_min_elements_;
  cv::Size min_roi_sz_;
  cv::Size max_roi_sz_;

  // Avg values
  float min_motion_per_pixel_;
  float min_motion_per_feature_;
  float min_optflow_per_feature_;

  float inf_factor_width_;
  float inf_factor_height_;
  std::size_t dbs_num_threads_;

  // The result of clustering
  obz::roi_map_t rois_map_;

  clustering::DBSCAN::ClusterData dbs_data_;
  clustering::DBSCAN dbs_engine_;

public:
  ROIExtraction();
  ROIExtraction(const double dbs_eps,
                const std::size_t dbs_min_elements,
                const cv::Size& min_roi_sz,
                const cv::Size& max_roi_sz,
                const float min_motion_per_pixel,
                const float min_motion_per_feature,
                const float min_optflow_per_feature,
                const float inflation_width,
                const float inflation_height,
                const std::size_t dbs_num_threads);

  ~ROIExtraction();

  bool Update(const obz::pts_vec_t& curr_tracked_features,
              const obz::pts_vec_t& prev_tracked_features,
              const cv::Mat& diff_frame);

  // This function appends data to bb_vec
  std::size_t GetValidBBs(obz::rect_vec_t& bb_vec, std::vector<float> &avg_flow) const;

  const obz::roi_map_t& GetROIsMap() const {return rois_map_;}

  void DrawROIs(cv::Mat& frame, const bool verbose = true);
};

}  // namespace obz

#endif

#ifndef ROI_EXTRACTION
#define ROI_EXTRACTIOn

#include "obzerver/benchmarker.hpp"
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
  double min_motion_per_pixel_;
  double inf_factor_width_;
  double inf_factor_height_;
  std::size_t dbs_num_threads_;

  // The result of clustering
  obz::roi_map_t rois_map_;

  clustering::DBSCAN::ClusterData dbs_data_;
  clustering::DBSCAN dbs_engine_;

  StepBenchmarker& ticker;
public:
  ROIExtraction();
  ROIExtraction(const double dbs_eps,
                const std::size_t dbs_min_elements,
                const cv::Size& min_roi_sz,
                const cv::Size& max_roi_sz,
                const double min_motion_per_pixel,
                const double inflation_width,
                const double inflation_height,
                const std::size_t dbs_num_threads = 1);

  ~ROIExtraction();

  bool Update(const obz::pts_vec_t& curr_features,
                     const obz::pts_vec_t& prev_features,
                     const cv::Mat& diff_frame);

  void DrawROIs(cv::Mat& frame);

  const obz::roi_map_t& GetROIsMap() const {return rois_map_;}
};

}  // namespace obz

#endif

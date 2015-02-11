#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include "smctc/smctc.hh"
#include "dbscan/dbscan.h"

#include "opencv2/core/core.hpp"
#include "obzerver/benchmarker.hpp"
#include "obzerver/tobject.hpp"
#include "obzerver/ccv_wrapper.hpp"

/* SMC Stuff */

namespace obz
{

struct smc_shared_param_t {
  cv::Mat obs_diff;
  cv::Mat obs_sof;
  cv::Mat camera_transform; // From t->t-1
  unsigned short int crop;
  std::size_t num_particles;
  double prob_random_move;
  double mm_displacement_stddev;
};

extern cv::Ptr<smc_shared_param_t> shared_data;

// It's very important that every element of this
// structure be of type double.
// because the total number of elements of
// smc::particle<particle_state_t> is being calculated
// assuming that all elements are of size double.
// creating a cv::Mat wrapper around particles needs
// homogenous elements in the mentioned data structure
struct particle_state_t {
  cv::Rect_<double> bb;
  bool recent_random_move;
};

double ParticleObservationUpdate(long t, const particle_state_t& X);
smc::particle<particle_state_t> ParticleInitialize(smc::rng* rng);
void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng* rng);

/* The Tracker */

enum tracking_status_t {
  TRACKING_STATUS_LOST = 0,
  TRACKING_STATUS_NEW,
  TRACKING_STATUS_TRACKING,
  TRACKING_STATUS_DISAPPEAR,
  TRACKING_STATUS_NUM
};

class PFObjectTracker {
protected:
  std::size_t num_particles;
  std::size_t hist_len;
  float fps;
  unsigned short int crop; //px
  double prob_random_move; //[0,1]
  double mm_displacement_noise_stddev; //px

  smc::sampler<particle_state_t> sampler;
  smc::moveset<particle_state_t> moveset;

  // Clustering and Object Tracking
  tracking_status_t status;
  std::size_t tracking_counter;
  unsigned short int num_clusters;
  cv::Mat labels;
  cv::Mat centers;
  std::vector<cv::Mat> vec_cov;
  double clustering_err_threshold;

  // DBSCAN
//  struct cluster_bb_t
//  {
//    cv::Rect bb;
//    bool is_object;
//    cluster_bb_t(): bb(cv::Rect(0,0,0,0)), is_object(false) {}
//    cluster_bb_t(const cv::Rect &bb_): bb(bb_), is_object(false) {}
//  };

  clustering::DBSCAN::ClusterData pts;
  clustering::DBSCAN dbs;
  std::map<std::int32_t, cv::Rect> motion_clusters_;
  std::vector<cv::Rect> object_bbs_;
  cv::Mat objectness_obs_;

  // ccv ICF object detection
  ccv::ICFCascadeClassifier* icf_classifier_;

  TObject tobject;

  StepBenchmarker& ticker;

  cv::Rect GenerateBoundingBox(const std::vector<cv::Point2f> &pts, const cv::Point2f center, const float cov_x, const float cov_y, const float alpha, const int max_width, const int boundary_width, const int boundary_height);
  cv::Rect GenerateBoundingBox(const std::vector<cv::Point2f> &pts, const std::vector<cv::Point2f> &weights, const float alpha, const float max_width, const int boundary_width, const int boundary_height);
public:
  PFObjectTracker(const std::size_t num_particles,
                const std::size_t hist_len,
                const float fps,
                ccv::ICFCascadeClassifier* icf_classifier,
                const unsigned short int crop = 30,
                const double prob_random_move = 0.25,
                const double mm_displacement_noise_stddev = 20);
  ~PFObjectTracker();

  bool Update(const cv::Mat& img_stab, const cv::Mat& img_diff, const cv::Mat& img_sof, const cv::Mat& camera_transform);

  bool Update2(const cv::Mat& img_stab, const cv::Mat& img_diff, const cv::Mat& camera_transform, const cv::Mat& img_rgb);

  const TObject& GetObject() const;
  cv::Rect GetObjectBoundingBox(std::size_t t = 0) const;
  tracking_status_t GetStatus() const {return status;}
  bool IsTracking() const {return status == TRACKING_STATUS_TRACKING;}
  void DrawParticles(cv::Mat& img);

};

}  // namespace obz
#endif


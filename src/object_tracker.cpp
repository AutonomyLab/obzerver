#include "glog/logging.h"

#include "obzerver/object_tracker.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/fft.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <map>

namespace obz
{

cv::Ptr<smc_shared_param_t> shared_data;

double ParticleObservationUpdate(long t, const particle_state_t &X)
{
  (void)t;  // shutup gcc
  int NN = 1; // Prevent div/0
  double corr_weight = 0.0;
//  const cv::Point2d vec2center = X.bb.tl() - cv::Point2d(
//        shared_data->obs_diff.cols/2.0, shared_data->obs_diff.rows/2.0);
//  const double dist2center = 1.0 - (((vec2center.x * vec2center.x) + (vec2center.y * vec2center.y)) / ((shared_data->obs_diff.cols * shared_data->obs_diff.cols) + (shared_data->obs_diff.rows * shared_data->obs_diff.rows)));
//  const double dist2center = 1.0;
  for (int i = -9; i < 10; i+=2) {
    for (int j = -9; j < 10; j+=2) {
      const int xx = (int) round(X.bb.tl().x) - i;
      const int yy = (int) round(X.bb.tl().y) - j;
      if (xx < shared_data->crop ||
          yy < shared_data->crop ||
          xx > (shared_data->obs_diff.cols - shared_data->crop) ||
          yy > (shared_data->obs_diff.rows - shared_data->crop))
      {
        continue;
      }
      NN++;
      corr_weight += pow(shared_data->obs_diff.ptr<uchar>(yy)[xx], 2);
      //corr_weight += shared_data->obs_diff.ptr<uchar>(yy)[xx];
      if (shared_data->obs_sof.data)
      {
        corr_weight += shared_data->obs_sof.ptr<float>(yy)[xx];
      }
    }
  }
//  double corr_weight = cv::mean(shared_data->obs_diff(bb))[0];
  //corr_weight = (fabs(corr_weight > 1e-6)) ? log(corr_weight/NN) : -13.0;
  corr_weight = (fabs(corr_weight > 1e-6)) ? (corr_weight/(float(NN) * (512.0 * 512))) : 1e-6;
//  LOG(INFO) << corr_weight;
  return log(corr_weight);
}

smc::particle<particle_state_t> ParticleInitialize(smc::rng *rng)
{
  particle_state_t p;
  p.bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
  p.bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.rows - shared_data->crop);
  p.recent_random_move = true;
  return smc::particle<particle_state_t>(p, -log(shared_data->num_particles)); // log(1/1000)
  //return smc::particle<particle_state_t>(p, ParticleObservationUpdate(0, p)); // log(1/1000)
}

void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng *rng)
{
  particle_state_t* cv_to = X.GetValuePointer();
  const double min_xy = shared_data->crop;
  const double max_x = shared_data->obs_diff.cols - shared_data->crop;
  const double max_y = shared_data->obs_diff.rows - shared_data->crop;
  if (rng->Uniform(0.0, 1.0) > shared_data->prob_random_move) {
    const cv::Point2f& p_stab = util::TransformPoint(cv_to->bb.tl(), shared_data->camera_transform.inv());
    cv_to->recent_random_move = false;
    cv_to->bb.x = util::clamp(rng->Normal(p_stab.x, shared_data->mm_displacement_stddev), min_xy, max_x);
    cv_to->bb.y = util::clamp(rng->Normal(p_stab.y, shared_data->mm_displacement_stddev), min_xy, max_y);
  } else {
    cv_to->recent_random_move = true;
    cv_to->bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
    cv_to->bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.rows - shared_data->crop);
  }
  //X.MultiplyWeightBy(ParticleObservationUpdate(t, *cv_to));
  X.AddToLogWeight(ParticleObservationUpdate(t, *cv_to));
}

// We are sharing img_diff and img_sof by reference (no copy)
PFObjectTracker::PFObjectTracker(const std::size_t num_particles,
                             const std::size_t hist_len,
                             const float fps,
                             ccv::ICFCascadeClassifier *icf_classifier,
                             const unsigned short int crop,
                             const double prob_random_move,
                             const double mm_displacement_noise_stddev)
  :
  num_particles(num_particles),
  hist_len(hist_len),
  fps(fps),
  crop(crop),
  prob_random_move(prob_random_move),
  mm_displacement_noise_stddev(mm_displacement_noise_stddev),
  sampler(num_particles, SMC_HISTORY_NONE),
  moveset(ParticleInitialize, ParticleMove),
  status(TRACKING_STATUS_LOST),
  tracking_counter(0),
  num_clusters(0),
  clustering_err_threshold(100),
  pts(num_particles, 2),
  dbs(0.01, 10, 1),
  icf_classifier_(icf_classifier),
  tobject(hist_len, fps),
  ticker(StepBenchmarker::GetInstance())
{
  shared_data = new smc_shared_param_t();
  shared_data->crop = crop;
  shared_data->num_particles = num_particles;
  shared_data->prob_random_move = prob_random_move;
  shared_data->mm_displacement_stddev = mm_displacement_noise_stddev;

  sampler.SetResampleParams(SMC_RESAMPLE_MULTINOMIAL, 0.99);
  //sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED, 2000);
  sampler.SetMoveSet(moveset);
  sampler.Initialise();

  LOG(INFO) << "[OT] Initialized object tracker with " << num_particles << " particles.";
  if (icf_classifier_ != 0)
  {
    LOG(INFO) << "[OT] Using cascade classifier: " << icf_classifier_->GetCascadeFile();
  }
  else
  {
    LOG(INFO) << "[OT] No cascade support";
  }
}

PFObjectTracker::~PFObjectTracker() {
  ;
}


bool PFObjectTracker::Update2(const cv::Mat& img_stab, const cv::Mat &img_diff, const cv::Mat &camera_transform, const cv::Mat& img_rgb)
{
  CV_Assert(img_diff.channels() == 1);
  // This is only shallow copy O(1)
  shared_data->obs_diff = img_diff;
  //shared_data->obs_sof = img_sof;
  shared_data->camera_transform = camera_transform;

  sampler.Iterate();
  ticker.tick("OT_Particle_Filter");

  for (long int i = 0; i < sampler.GetNumber(); i++)
  {
    const particle_state_t& p = sampler.GetParticleValue(i);
    pts(i, 0) = -1.0 + (p.bb.x / 480.0);
    pts(i, 1) = -1.0 + (p.bb.y / 270.0);
  }
  ticker.tick("OT_Particles_to_Mat");
//  LOG(INFO) << "Size of matrix is: " << pts.size1() << " x " << pts.size2();

  dbs.init(0.03, num_particles / 20, 4);
  dbs.reset();
  dbs.fit(pts);
  ticker.tick("OT_Clustering");

  CV_Assert(dbs.get_labels().size() == num_particles);

  const clustering::DBSCAN::Labels& cluster_labels = dbs.get_labels();
  num_clusters = 0;
  motion_clusters_.clear();
  for (std::size_t i = 0; i < cluster_labels.size(); i++)
  {
    const std::int32_t cid = cluster_labels[i];
    const particle_state_t& p = sampler.GetParticleValue(i);

    if (cid == -1) continue;
    if (motion_clusters_.count(cid) == 0)
    {
      motion_clusters_.insert(std::pair<int32_t, cv::Rect>(cid, cv::Rect(p.bb.x, p.bb.y, 1, 1)));
      num_clusters++;
    }
    cv::Rect& r = motion_clusters_[cid];

    if (p.bb.x < r.x)  // grow left
    {
      r.x = p.bb.x;
      r.width = std::max(r.br().x - r.x, 1);
    }
    else if (p.bb.x > r.br().x) // grow right
    {
      r.width = std::max(static_cast<std::int32_t>(p.bb.x) - r.x, 1);
    }

    if (p.bb.y < r.y)  // grow top
    {
      r.y = p.bb.y;
      r.height = std::max(r.br().y - r.y, 1);
    }
    else if (p.bb.y > r.br().y) // grow bottom
    {
      r.height = std::max(static_cast<std::int32_t>(p.bb.y) - r.y, 1);
    }
    CV_Assert(r.width >=0 && r.height >= 0);
    CV_Assert(r.x >= 0 && r.y >=0 && r.br().x < img_stab.cols && r.br().y < img_stab.rows);

//    r.x = std::min(static_cast<std::int32_t>(round(p.bb.tl().x)), r.x);
//    r.width = std::max(static_cast<std::int32_t>(round(p.bb.br().x)), r.x) - r.x;
//    r.y = std::min(static_cast<std::int32_t>(round(p.bb.tl().y)), r.y);
//    r.height = std::max(static_cast<std::int32_t>(round(p.bb.br().y)), r.y) - r.y;
  }

  CV_Assert(num_clusters == motion_clusters_.size());
  ticker.tick("OT_Clustering_BB");
  LOG(INFO) << "Number of clusters: " << num_clusters;

  object_bbs_.clear();
  if (icf_classifier_)
  {
    objectness_obs_ = cv::Mat::zeros(img_stab.rows, img_stab.cols, CV_32FC1);
    std::vector<ccv::ICFCascadeClassifier::result_t> icf_result_vec;
    for (std::size_t i = 0; i < num_clusters; i++)
    {
      if (motion_clusters_[i].width == 0 || motion_clusters_[i].height == 0)
      {
        LOG(WARNING) << "Skipping";
        continue;
      }
      icf_result_vec.clear();
      const std::size_t num_det_objects = icf_classifier_->Detect(img_rgb, icf_result_vec, motion_clusters_[i]);
      for (std::size_t j = 0; j < num_det_objects; j++)
      {
        object_bbs_.push_back(icf_result_vec[i].bb);
        cv::rectangle(objectness_obs_, icf_result_vec[i].bb, cv::Scalar(1e2, 1e2, 1e2), CV_FILLED);
      }
      if (!num_det_objects)
      {
        cv::rectangle(objectness_obs_, motion_clusters_[i], cv::Scalar(-5, -5, -5), CV_FILLED);
      }
      LOG(INFO) << "Cluster #" << i << " : " << icf_result_vec.size();
    }
    shared_data->obs_sof = objectness_obs_;  // shallow
    ticker.tick("OT_ICF");
  }

  // Let's filter out the BBs using object detection
  tobject.Reset();
  status = TRACKING_STATUS_LOST;
  return false;
}

bool PFObjectTracker::Update(const cv::Mat& img_stab, const cv::Mat &img_diff, const cv::Mat &img_sof, const cv::Mat &camera_transform)
{
  CV_Assert(img_diff.channels() == 1);
  //CV_Assert(img_sof.type() == CV_8UC1);

  // This is only shallow copy O(1)
  shared_data->obs_diff = img_diff;
  shared_data->obs_sof = img_sof;
  shared_data->camera_transform = camera_transform;

  sampler.Iterate();
  ticker.tick("OT_Particle_Filter");

  // to use this method smctc/sampler.hh needs to be pathced to
  // return the pParticles pointer
  // particle<Space>* GetParticlesPtr() {return pParticles; }
  // No copying, just a wrapper, do not mess with this matrix.
  // Use it only as a readonly
  // bb.x bb.y bb.w bb.h logweight
//  cv::Mat particles(sampler.GetNumber(),
//               sizeof(smc::particle<particle_state_t>) / sizeof(double),
//               CV_64FC1, sampler.GetParticlesPtr());
//  cv::Mat particles_pose;
//  particles.colRange(0, 2).convertTo(particles_pose, CV_32F);


  //std::vector<cv::Point2f> pts(sampler.GetNumber());

  std::vector<cv::Point2f> pts;
  std::vector<cv::Point2f> pts_w;
  for (long i = 0; i < sampler.GetNumber(); i++) {
    const particle_state_t& p = sampler.GetParticleValue(i);
    const double w = sampler.GetParticleLogWeight(i);
    if (false == p.recent_random_move) {
      //pts[i] = p.bb.tl();
      pts.push_back(p.bb.tl());
      pts_w.push_back(cv::Point2f(w, w));
      LOG(INFO) << p.bb.tl().x << "," << p.bb.tl().y << "," << w;
    }
  }

  LOG(INFO) << "[OT] Number of particles to consider: " << pts.size();
  if (pts.size() < 0.5 * sampler.GetNumber()) {
    LOG(WARNING) << "[OT] Number of non-random partilces are not enough for clustering ...";
    return false;
  }
  cv::Mat particles_pose(pts, false);
  particles_pose = particles_pose.reshape(1);

  ticker.tick("OT_Particles_to_Mat");

//  cv::Mat particles_mask = cv::Mat::zeros(img_stab.size(), CV_8UC1);
//  for (std::size_t i = 0; i < pts.size(); i++) {
//    cv::rectangle(particles_mask, cv::Rect(pts[i], cv::Size(5,5)), cv::Scalar(255, 255, 255));
//  }
//  cv::bitwise_and(img_stab, particles_mask, particles_mask);
//  ticker.tick("  [OT] Particles Mask");

  unsigned short int k = 0;
  double err = 1e12;
  vec_cov.clear();
  // 4 is the maximum number of clusters
  for (k = 1; k < 4 && err > clustering_err_threshold; k++) {
    cv::EM em_estimator(k,
                        cv::EM::COV_MAT_DIAGONAL,
                        cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01));

    em_estimator.train(particles_pose, cv::noArray(), labels, cv::noArray());

    centers = em_estimator.get<cv::Mat>("means");
    vec_cov = em_estimator.get<std::vector<cv::Mat> >("covs");
    err = 0.0;
    // Is vec_cov.size() == k??
    for (std::size_t c = 0; c < vec_cov.size(); c++) {
      err += (sqrt(vec_cov[c].at<double>(0,0)) + sqrt(vec_cov[c].at<double>(1,1)));
//      LOG(INFO) << "-- " << c << " : " << centers.row(c) << " " << err;//vec_cov[c];
    }
    err /= double(k);
    LOG(INFO) << ">>>> #" << k << " : " << err;
//    err = cv::kmeans(particles_pose, // Only on positions
//                     k,
//                     labels,
//                     cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01)
//                     , 20, cv::KMEANS_PP_CENTERS, centers);
//    LOG(INFO) << "Tried with: " << k << " Actuall Err:" << sqrt(err);
//    err = sqrt(err / (double) particles_pose.rows);
//    LOG(INFO) << "Tried with: " << k << " Err:" << err;
  }
  if (err <= clustering_err_threshold) {
    num_clusters = k - 1;
    LOG(INFO) << "[OT]  Particles: " << particles_pose.rows << " Number of clusers in particles: " << num_clusters << " Centers" << centers << " err: " << err;
  } else {
    num_clusters = 0;
    LOG(WARNING) << "[OT]  Clustering failed. Particles are very sparse " << err;
  }

  ticker.tick("OT_Clustering");

  if (status == TRACKING_STATUS_LOST) {
    if (num_clusters != 1) {
      LOG(INFO) << "[OT]  Waiting for first dense cluser ...";
    } else {
      // All pts are condensed enough to form a bounding box     
      // MANI
      cv::Rect new_bb;
//      cv::Rect new_bb = GenerateBoundingBox(pts,
//                                            cv::Point2f(
//                                              centers.at<double>(0,0),
//                                              centers.at<double>(0, 1)),
//                                            vec_cov[0].at<double>(0,0),
//                                            vec_cov[0].at<double>(1,1),
//                                            5.0, 100.0, img_diff.cols, img_diff.rows);
      if (new_bb.area() > 0) {
        tobject.Update(new_bb, img_stab, true);
        status = TRACKING_STATUS_TRACKING;
        tracking_counter = 15;
      } else {
        LOG(WARNING) << "[OT] Initial BB has area of size 0. Skipping.";
      }
    }
  } else if (status == TRACKING_STATUS_TRACKING) {
    cv::Rect tracked_bb = tobject().latest().bb;
    const cv::Point2f& p_stab = util::TransformPoint(tracked_bb.tl(), camera_transform.inv());
    tracked_bb.x = p_stab.x;
    tracked_bb.y = p_stab.y;
    if (tracking_counter == 0) {
      LOG(WARNING) << "[OT] Lost Track";
      tobject.Reset();
      status = TRACKING_STATUS_LOST;
    } else {
      double min_dist = 1e12;
      cv::Rect bb;
      cv::Mat fg; //foreground mask
      std::vector<cv::Point2f> cluster;
      std::vector<cv::Point2f> cluster_w;
      int min_dist_cluster = 0;
      for (unsigned int i = 0; i < num_clusters; i++) {
        const cv::Point2f pt(centers.at<double>(i, 0), centers.at<double>(i, 1));
        double dist = pow(pt.x - util::RectCenter(tracked_bb).x, 2);
        dist += pow(pt.y - util::RectCenter(tracked_bb).y, 2);
        dist = sqrt(dist);
        //LOG(INFO) << "[OT] Distance frome " << rectCenter(tracked_bb) << " to " << pt << " is " << dist << std::endl;
        if (dist < min_dist) {
          min_dist = dist;
          min_dist_cluster = i;
        }
      }
      LOG(INFO) << "[OT] Chose cluster #" << min_dist_cluster << " with d = " << min_dist;
      if (num_clusters == 0) {
        LOG(WARNING) << "[OT] No cluster at all. Skipping.";
        bb = tracked_bb;
        tracking_counter--;
      }
      if  (min_dist < 200) {
        // Mani
//        for (unsigned int i = 0; i < pts.size1(); i++) {
//            if (labels.at<int>(i) == min_dist_cluster) {
//                cluster.push_back(pts.at(i));
//                cluster_w.push_back(pts_w.at(i));
//            }
//        }
        //LOG(INFO) << "Subset size: " << cluster.size() << std::endl;
        //LOG(INFO) << "Coovariance: " << vec_cov[min_dist_cluster];

        // MANI

//        bb = GenerateBoundingBox(cluster,
//                                 cv::Point2f(
//                                   centers.at<double>(min_dist_cluster,0),
//                                   centers.at<double>(min_dist_cluster, 1)),
//                                 vec_cov[min_dist_cluster].at<double>(0,0),
//                                 vec_cov[min_dist_cluster].at<double>(1,1),
//                                 5.0, 100, img_diff.cols, img_diff.rows);

        tracking_counter = 15;
      } else {
        LOG(INFO) << "[OT] The closest cluster is far from current object being tracked, skipping";
        bb = tracked_bb;
        tracking_counter--;
      }
//      LOG(INFO) << " BB: " << bb;
      const double param_lpf = 0.5;
      tracked_bb.x = param_lpf * tracked_bb.x + (1.0 - param_lpf) * bb.x;
      tracked_bb.y = param_lpf * tracked_bb.y + (1.0 - param_lpf) * bb.y;
      tracked_bb.width = param_lpf * tracked_bb.width + (1.0 - param_lpf) * bb.width;
      tracked_bb.height = param_lpf * tracked_bb.height + (1.0 - param_lpf) * bb.height;
      tracked_bb = util::ClampRect(tracked_bb, img_stab.cols, img_stab.rows);
      if (tracked_bb.area() == 0) {
        // 0-tolerance!
        LOG(WARNING) << "Tracked BB Area is 0. Resetting.";
        tobject.Reset();
        status = TRACKING_STATUS_LOST;
      } else {
        tobject.Update(tracked_bb, img_stab);
      }
    }
  }

  ticker.tick("OT_Tracking");
  LOG(INFO) << "[OT] Tracking Status: " << status  << " Tracked: " << GetObjectBoundingBox();
  return true;
}

cv::Rect PFObjectTracker::GenerateBoundingBox(const std::vector<cv::Point2f> &pts,
                                            const cv::Point2f center,
                                            const float cov_x,
                                            const float cov_y,
                                            const float alpha,
                                            const int max_width,
                                            const int boundary_width,
                                            const int boundary_height)
{
  // 3 * std_dev thresshold for outliers
  double cov_xt = 9.0 * cov_x;
  double cov_yt = 9.0 * cov_y;
  std::vector<cv::Point2f> pts_inliers;
  for (unsigned int i = 0; i < pts.size(); i++) {
    if (pow(pts[i].x - center.x, 2.0) < cov_xt &&
        pow(pts[i].y - center.y, 2.0) < cov_yt)
    {
      pts_inliers.push_back(pts[i]);
    }
  }

  //LOG(INFO) << "Pruning: " << pts_inliers.size() << " / " << pts.size();
  cv::Scalar _c, _s;
  cv::meanStdDev(pts_inliers, _c, _s);
  const double _w = std::min(_s[0] * alpha, (double) max_width);
  const double _h = std::min(_s[1] * alpha, (double) max_width);

  cv::Rect bb
      (
        int(_c[0] - _w/2.0),
        int(_c[1] - _h/2.0),
        int(_w),
        int(_h)
      );

  //LOG(INFO) << "Generate BB: " << _c[0] << " " << _c[1] << " " << _s[0] <<  " " << _s[1];
  return util::ClampRect(bb, boundary_width, boundary_height);
}

cv::Rect PFObjectTracker::GenerateBoundingBox(const std::vector<cv::Point2f>& pts,
                                            const std::vector<cv::Point2f>& weights,
                                            const float alpha,
                                            const float max_width,
                                            const int boundary_width,
                                            const int boundary_height)
{
  // TODO: Fix the variance bias
  cv::Mat pts_mat(pts, false);
  cv::Mat w_mat(weights, false);
  cv::Scalar _c, _s;
  double sum_w = cv::sum(w_mat.col(0))[0];
  double sum_w_sqr = cv::sum(w_mat.col(0).mul(w_mat.col(0)))[0];
  LOG(INFO) << "sum w: " << sum_w << " sum w^2 " << sum_w_sqr;
  cv::meanStdDev(pts_mat.mul(w_mat), _c, _s);
  LOG(INFO) << _c[0] << " " << _c[1] << " " << _s[0] <<  " " << _s[1];
  _c[0] *= (float(pts.size()) / sum_w);
  _c[1] *= (float(pts.size()) / sum_w);
//  const double s_w = sum_w / ((sum_w*sum_w) - sum_w_sqr);
  _s[0] *= (float(pts.size()) / sum_w);
  _s[1] *= (float(pts.size()) / sum_w);
  LOG(INFO) << _c[0] << " " << _c[1] << " " << _s[0] <<  " " << _s[1];
  float _e = std::min((float) (2.0 * alpha * std::max(_s[0], _s[1])), max_width);
  cv::Rect bb(int(_c[0] - _e/2.0), int(_c[1] - _e/2.0), _e, _e);

  // TODO: fix ratio
  return util::ClampRect(bb, boundary_width, boundary_height);
}

const TObject &PFObjectTracker::GetObject() const
{
  return tobject;
}

cv::Rect PFObjectTracker::GetObjectBoundingBox(std::size_t t) const {
  return (tobject().size()) ? tobject.Get(t).bb : cv::Rect(0, 0, 0, 0);
}

/* Debug */

void PFObjectTracker::DrawParticles(cv::Mat &img)
{
  for (long i = 0; i < sampler.GetNumber(); i++) {
    if (!sampler.GetParticleValue(i).recent_random_move)
    {
      cv::circle(img, sampler.GetParticleValue(i).bb.tl(), std::max(0.0, sampler.GetParticleWeight(i)), cv::Scalar(0, 0, 0), -1);
    }
    else
    {
      cv::circle(img, sampler.GetParticleValue(i).bb.tl(), std::max(0.0, sampler.GetParticleWeight(i)), cv::Scalar(255, 255, 255), -1);
    }
  }

  for (auto &bb: motion_clusters_)
  {
    LOG(INFO) << bb.first << " : " << bb.second;
    cv::rectangle(img, bb.second, cv::Scalar(255, 255, 255));
  }

  for (auto &bb: object_bbs_)
  {
    cv::rectangle(img, bb, cv::Scalar(50, 50, 50));
  }
//  for (int i = 0; i < num_clusters; i++) {
//    const cv::Point2f pt(centers.at<double>(i,0), centers.at<double>(i,1));

//    const double err = sqrt(vec_cov[i].at<double>(0,0)) + sqrt(vec_cov[i].at<double>(1,1));
//    cv::circle(img, pt, err, cv::Scalar(255, 255, 255));
//    //cv::circle(img, centers.at<cv::Point2f>(i), 10, cv::Scalar(255, 255, 255));
//  }

  if (status != TRACKING_STATUS_LOST) {
    cv::rectangle(img, GetObjectBoundingBox(), cv::Scalar(127, 127, 127), 5);
  }
}

}  // namespace obz

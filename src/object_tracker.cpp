#include "glog/logging.h"

#include "obzerver/object_tracker.hpp"
#include "obzerver/utility.hpp"
#include "obzerver/fft.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

cv::Ptr<smc_shared_param_t> shared_data;

double ParticleObservationUpdate(long t, const particle_state_t &X)
{
  int NN = 1; // Prevent div/0
  double corr_weight = 0.0;
  for (int i = -9; i < 10; i+=2) {
    for (int j = -9; j < 10; j+=2) {
      int xx = (int) round(X.bb.tl().x) - i;
      int yy = (int) round(X.bb.tl().y) - j;
      if (xx < shared_data->crop ||
          yy < shared_data->crop ||
          xx > (shared_data->obs_diff.cols - shared_data->crop) ||
          yy > (shared_data->obs_diff.rows - shared_data->crop))
      {
        continue;
      }
      NN++;
      corr_weight += shared_data->obs_diff.ptr<uchar>(yy)[xx];
    }
  }
//  double corr_weight = cv::mean(shared_data->obs_diff(bb))[0];
  //corr_weight = (fabs(corr_weight > 1e-6)) ? log(corr_weight/NN) : -13.0;
  corr_weight = (fabs(corr_weight > 1e-6)) ? (corr_weight/(float(NN) * 512.0)) : 1e-6;
  //OG(INFO) << corr_weight;
  return corr_weight;
}

smc::particle<particle_state_t> ParticleInitialize(smc::rng *rng)
{
  particle_state_t p;
  p.bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
  p.bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.rows - shared_data->crop);
  p.recent_random_move = true;
  //return smc::particle<particle_state_t>(p, -log(shared_data->num_particles)); // log(1/1000)
  return smc::particle<particle_state_t>(p, ParticleObservationUpdate(0, p)); // log(1/1000)
}

void ParticleMove(long t, smc::particle<particle_state_t> &X, smc::rng *rng)
{
  particle_state_t* cv_to = X.GetValuePointer();
  if (rng->Uniform(0.0, 1.0) > shared_data->prob_random_move) {
    cv::Point2f p_stab = transformPoint(cv_to->bb.tl(), shared_data->camera_transform.inv());
    cv_to->recent_random_move = false;
    cv_to->bb.x = p_stab.x + rng->Normal(0, shared_data->mm_displacement_stddev);
    cv_to->bb.y = p_stab.y + rng->Normal(0, shared_data->mm_displacement_stddev);
  } else {
    cv_to->recent_random_move = true;
    cv_to->bb.x = rng->Uniform(shared_data->crop, shared_data->obs_diff.cols - shared_data->crop);
    cv_to->bb.y = rng->Uniform(shared_data->crop, shared_data->obs_diff.rows - shared_data->crop);
  }
  X.MultiplyWeightBy(ParticleObservationUpdate(t, *cv_to));
  //X.AddToLogWeight(ParticleObservationUpdate(t, *cv_to));

}

// We are sharing img_diff and img_sof by reference (no copy)
ObjectTracker::ObjectTracker(const std::size_t num_particles,
                             const std::size_t hist_len,
                             const float fps,
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
  tobject(hist_len, fps),
  ticker(StepBenchmarker::GetInstance())
{
  shared_data = new smc_shared_param_t();
  shared_data->crop = crop;
  shared_data->num_particles = num_particles;
  shared_data->prob_random_move = prob_random_move;
  shared_data->mm_displacement_stddev = mm_displacement_noise_stddev;

  //sampler.SetResampleParams(SMC_RESAMPLE_SYSTEMATIC, 0.5);
  sampler.SetResampleParams(SMC_RESAMPLE_STRATIFIED, 2000);
  sampler.SetMoveSet(moveset);
  sampler.Initialise();

  LOG(INFO) << "Initialized object tracker with " << num_particles << " particles.";
}

ObjectTracker::~ObjectTracker() {
  LOG(INFO) << "Object tracker destroyed.";
}

bool ObjectTracker::Update(const cv::Mat& img_stab, const cv::Mat &img_diff, const cv::Mat &img_sof, const cv::Mat &camera_transform)
{
  CV_Assert(img_diff.channels() == 1);
  //CV_Assert(img_sof.type() == CV_8UC1);

  // This is only shallow copy O(1)
  shared_data->obs_diff = img_diff;
  shared_data->obs_sof = img_sof;
  shared_data->camera_transform = camera_transform;

  sampler.Iterate();
  ticker.tick("  [OT] Particle Filter");

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
    const particle_state_t p = sampler.GetParticleValue(i);
    const double w = sampler.GetParticleWeight(i);
    if (false == p.recent_random_move) {
      //pts[i] = p.bb.tl();
      pts.push_back(p.bb.tl());
      pts_w.push_back(cv::Point2f(w, w));
      //LOG(INFO) << i << " " << p.bb << " " << w << " " << pts.back();
    }
  }

  LOG(INFO) << "Number of particles to consider: " << pts.size();
  if (pts.size() < 0.5 * sampler.GetNumber()) {
    LOG(WARNING) << "Number of non-random partilces are not enough for clustering ...";
    return false;
  }
  cv::Mat particles_pose(pts, false);
  particles_pose = particles_pose.reshape(1);

  ticker.tick("  [OT] Particles -> Mat");

//  cv::Mat particles_mask = cv::Mat::zeros(img_stab.size(), CV_8UC1);
//  for (std::size_t i = 0; i < pts.size(); i++) {
//    cv::rectangle(particles_mask, cv::Rect(pts[i], cv::Size(5,5)), cv::Scalar(255, 255, 255));
//  }
//  cv::bitwise_and(img_stab, particles_mask, particles_mask);
//  ticker.tick("  [OT] Particles Mask");

  unsigned short int k = 0;
  double err = 1e12;
  std::vector<cv::Mat> vec_cov;
  for (k = 1; k < 5 && err > clustering_err_threshold; k++) {
    cv::EM em_estimator(k, cv::EM::COV_MAT_DIAGONAL, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01));
    em_estimator.train(particles_pose, cv::noArray(), labels, cv::noArray());

    centers = em_estimator.get<cv::Mat>("means");
    vec_cov = em_estimator.get<std::vector<cv::Mat> >("covs");
    LOG(INFO) << "EM k: " << k;
    err = 0.0;
    for (std::size_t c = 0; c < vec_cov.size(); c++) {
      err += (sqrt(vec_cov[c].at<double>(0,0)) + sqrt(vec_cov[c].at<double>(1,1)));
      LOG(INFO) << "-- " << c << " : " << centers.row(c) << " " << err;//vec_cov[c];
    }
    err /= double(k);
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
    LOG(INFO) << "Particles: "<< particles_pose.rows << " Number of clusers in particles: " << num_clusters << " Centers" << centers << " err: " << err;
  } else {
    num_clusters = 0;
    LOG(WARNING) << "Clustering failed. Particles are very sparse.";
  }

  ticker.tick("  [OT] Clustering");

  if (status == TRACKING_STATUS_LOST) {
    if (num_clusters != 1) {
      LOG(INFO) << "Waiting for first dense cluser ...";
    } else {
      // All pts are condensed enough to form a bounding box
      LOG(INFO) << "Coovariance: " << vec_cov[0];
      tobject.Update(img_stab, GenerateBoundingBox(pts,
                                                   cv::Point2f(centers.at<double>(0,0), centers.at<double>(0, 1)),
                                                   vec_cov[0].at<double>(0,0),
                                                   vec_cov[0].at<double>(1,1),
                                                   5.0, 100.0, img_diff.cols, img_diff.rows), true);
      status = TRACKING_STATUS_TRACKING;
      tracking_counter = 15;
    }
  } else if (status == TRACKING_STATUS_TRACKING) {
    cv::Rect tracked_bb = tobject().latest().bb;
    cv::Point2f p_stab = transformPoint(tracked_bb.tl(), camera_transform.inv());
    tracked_bb.x = p_stab.x;
    tracked_bb.y = p_stab.y;
    if (tracking_counter == 0) {
      LOG(INFO) << "Lost Track";
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
        double dist = pow(pt.x - rectCenter(tracked_bb).x, 2);
        dist += pow(pt.y - rectCenter(tracked_bb).y, 2);
        dist = sqrt(dist);
        LOG(INFO) << "Distance frome " << rectCenter(tracked_bb) << " to " << pt << " is " << dist << std::endl;
        if (dist < min_dist) {
          min_dist = dist;
          min_dist_cluster = i;
        }
      }
      LOG(INFO) << "Chose cluster #" << min_dist_cluster << " with d = " << min_dist;
      if (num_clusters == 0) {
        LOG(INFO) << "No cluster at all. Skipping.";
        bb = tracked_bb;
        tracking_counter--;
      }
      if  (min_dist < 200) {
        for (unsigned int i = 0; i < pts.size(); i++) {
            if (labels.at<int>(i) == min_dist_cluster) {
                cluster.push_back(pts.at(i));
                cluster_w.push_back(pts_w.at(i));
            }
        }
        LOG(INFO) << "Subset size: " << cluster.size() << std::endl;
        LOG(INFO) << "Coovariance: " << vec_cov[min_dist_cluster];
        bb = GenerateBoundingBox(cluster,
                                 cv::Point2f(centers.at<double>(min_dist_cluster,0), centers.at<double>(min_dist_cluster, 1)),
                                 vec_cov[min_dist_cluster].at<double>(0,0),
                                 vec_cov[min_dist_cluster].at<double>(1,1),
                                 5.0, 100, img_diff.cols, img_diff.rows);

        tracking_counter = 15;
      } else {
        LOG(INFO) << "The closest cluster is far from current object being tracked, skipping";
        bb = tracked_bb;
        tracking_counter--;
      }
      LOG(INFO) << " BB: " << bb;
      const double param_lpf = 0.5;
      tracked_bb.x = param_lpf * tracked_bb.x + (1.0 - param_lpf) * bb.x;
      tracked_bb.y = param_lpf * tracked_bb.y + (1.0 - param_lpf) * bb.y;
      tracked_bb.width = param_lpf * tracked_bb.width + (1.0 - param_lpf) * bb.width;
      tracked_bb.height = param_lpf * tracked_bb.height + (1.0 - param_lpf) * bb.height;
      tracked_bb = ClampRect(tracked_bb, img_stab.cols, img_stab.rows);
      tobject.Update(img_stab, tracked_bb);
    }
  }

  ticker.tick("  [OT] Tracking");
  LOG(INFO) << "Tracking Status: " << status  << " Tracked: " << GetObjectBoundingBox();
  return true;
}

cv::Rect ObjectTracker::GenerateBoundingBox(const std::vector<cv::Point2f> &pts,
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

  LOG(INFO) << "Pruning: " << pts_inliers.size() << " / " << pts.size();
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

  LOG(INFO) << "Generate BB: " << _c[0] << " " << _c[1] << " " << _s[0] <<  " " << _s[1];
  return ClampRect(bb, boundary_width, boundary_height);
}

cv::Rect ObjectTracker::GenerateBoundingBox(const std::vector<cv::Point2f>& pts,
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
  return ClampRect(bb, boundary_width, boundary_height);
}

const TObject &ObjectTracker::GetObject() const
{
  return tobject;
}

cv::Rect ObjectTracker::GetObjectBoundingBox(std::size_t t) const {
  return (tobject().size()) ? tobject.Get(t).bb : cv::Rect(0, 0, 0, 0);
}

/* Debug */

void ObjectTracker::DrawParticles(cv::Mat &img)
{
  for (long i = 0; i < sampler.GetNumber(); i++) {
    if (!sampler.GetParticleValue(i).recent_random_move)
      cv::circle(img, sampler.GetParticleValue(i).bb.tl(), std::max(0.0, sampler.GetParticleWeight(i)), cv::Scalar(0, 0, 0), -1);
  }

  for (int i = 0; i < num_clusters; i++) {
    const cv::Point2f pt(centers.at<double>(i,0), centers.at<double>(i,1));
    cv::circle(img, pt, 10, cv::Scalar(255, 255, 255));
    //cv::circle(img, centers.at<cv::Point2f>(i), 10, cv::Scalar(255, 255, 255));
  }

  if (status != TRACKING_STATUS_LOST) {
    cv::rectangle(img, GetObjectBoundingBox(), cv::Scalar(127, 127, 127), 5);
  }
}

#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
      
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  Hj_ << 1, 0, 0, 0,
  		 0, 1, 0, 0,
         0, 0, 1, 0;

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  // the state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 1,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) 
  {
    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
      // Convert radar from polar to cartesian coordinates 
      // and initialize state.
      float px = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
      float py = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);
      float vx = measurement_pack.raw_measurements_[2] * cos(measurement_pack.raw_measurements_[1]);
      float vy = measurement_pack.raw_measurements_[2] * sin(measurement_pack.raw_measurements_[1]);
      ekf_.x_ << px, 
      		   py, 
      		   vx, 
      		   vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
    {
      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
      		   measurement_pack.raw_measurements_[1],
      		   0,
      		   0;
    }
    
	previous_timestamp_ = measurement_pack.timestamp_;
  
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  double noise_ax = 9;
  double noise_ay = 9;
  
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
         	  0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
         	  dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
         	  0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  ekf_.Predict();

  /**
   * Update
   */
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  {
    // Radar updates
    VectorXd input(4);
	input << ekf_.x_(0), ekf_.x_(1), ekf_.x_(2), ekf_.x_(3);
    Hj_ = tools.CalculateJacobian(input);
    
    ekf_.H_ = MatrixXd(3, 4);
    ekf_.H_ = Hj_;
    ekf_.R_ = MatrixXd(3, 3);
    ekf_.R_ = R_radar_;
    
    VectorXd z(3);
    z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], measurement_pack.raw_measurements_[2];
    ekf_.UpdateEKF(z);
  } 
  else 
  {
    // Laser updates
    ekf_.H_ = MatrixXd(2, 4);
    ekf_.H_ = H_laser_;
    ekf_.R_ = MatrixXd(2, 2);
    ekf_.R_ = R_laser_;   
    
    VectorXd z(2);
	z << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1];
    ekf_.Update(z);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}

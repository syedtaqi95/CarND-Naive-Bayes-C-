#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include "Eigen/Dense"

using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB(int X_size);

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data, 
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

  vector<string> possible_labels = {"left","keep","right"};

  // Store the mean, stddev and M2 for each feature/label pair
  // Format is means[label][feature]
  vector<vector<double>> means;
  vector<vector<double>> variances;
  vector<vector<double>> M2;

  // Store the prior probability for each label
  vector<double> priors;
};

#endif  // CLASSIFIER_H
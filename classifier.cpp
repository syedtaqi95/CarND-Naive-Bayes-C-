#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>
#include <iostream>

using Eigen::ArrayXd;
using std::string;
using std::vector;
using std::cout;
using std::endl;

// Initializes GNB
GNB::GNB(int X_size) {
  /**
   * Done: Initialize GNB, if necessary. May depend on your implementation.
   */

  // Initialise means, variances and M2 to 0
  for (int label = 0; label < possible_labels.size(); label++) {
    vector<double> temp_vec;
    for(int f = 0; f < X_size; f++) {
      temp_vec.push_back(0.0);
    }
    means.push_back(temp_vec);
    variances.push_back(temp_vec);
    M2.push_back(temp_vec);
  }  
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * TODO: Implement the training function for your classifier.
   */

  // Store how many times each label occurs in the training data
  vector<int> label_count = {0, 0, 0};

  for (int sample = 0; sample < labels.size(); sample++) {
    for (int label = 0; label < possible_labels.size(); label++) {
      if(labels[sample] == possible_labels[label]) {
        // increment the running label count
        label_count[label] += 1;

        // Calculate the running mean and stddev of each feature/label pair
        // Uses Welford algorithm
        for (int feature = 0; feature < data[0].size(); feature++) {
          double delta = data[sample][feature] - means[label][feature];
          means[label][feature] += delta / double(label_count[label] + 1);
          M2[label][feature] += delta * (data[sample][feature] - means[label][feature]);
          variances[label][feature] = M2[label][feature] / double(label_count[label] + 1);
        }
      }
    }
  }

  // Calculate the prior probability for each label
  for (int i = 0; i < possible_labels.size(); i++) {
    double prior_value = double(label_count[i]) / double(labels.size());
    priors.push_back(prior_value);
  }

  // Debug
  /*cout << "priors = " << priors[0] << " " << priors[1] << " " << priors[2] << endl;
  cout << "means = " << endl;
  for (auto i = 0; i < means.size(); i++) {
    for (auto j = 0; j < means[i].size(); j++) {
      cout << means[i][j] << ' ';
    }
    cout << endl;
  }
  */
}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * TODO: Complete this function to return your classifier's prediction
   */
  
  return this -> possible_labels[1];
}
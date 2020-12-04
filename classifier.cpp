#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>

using Eigen::ArrayXd;
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::begin;
using std::end;

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
  vector<int> label_count;

  // Init to 0
  for (int label = 0; label < possible_labels.size(); label++) {
    label_count.push_back(0);
  }

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

  // Debug prints
  cout << "-------------" << endl;
  cout << "priors = " << priors[0] << " " << priors[1] << " " << priors[2] << endl;
  
  cout << "means = " << endl;
  for (auto i = 0; i < means.size(); i++) {
    for (auto j = 0; j < means[i].size(); j++) {
      cout << means[i][j] << ' ';
    }
    cout << endl;
  }

  cout << "variances = " << endl;
  for (auto i = 0; i < variances.size(); i++) {
    for (auto j = 0; j < variances[i].size(); j++) {
      cout << variances[i][j] << ' ';
    }
    cout << endl;
  }
  cout << "-------------" << endl;  
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

  // Vector of prob density functions for each feature/label pair
  vector<vector<double>> pdf;

  // Init all pdf's to 0
  for (int label = 0; label < possible_labels.size(); label++) {
    vector<double> temp_vec;
    for(int f = 0; f < sample.size(); f++) {
      temp_vec.push_back(0.0);
    }
    pdf.push_back(temp_vec);
  }

  // Calculate Gaussian pdf's
  for (int label = 0; label < possible_labels.size(); label++) {
    for (int feature = 0; feature < sample.size(); feature++) {      
      double norm_term = 1 / (sqrt(2*M_PI*variances[label][feature]));
      double exp_term = - pow(sample[feature] - means[label][feature], 2) / (2*variances[label][feature]);
      pdf[label][feature] = norm_term * exp(exp_term);
    }
  }

  // Calculate conditional probabilities for each label
  vector<double> cond_probs;
  for (int label = 0; label < possible_labels.size(); label++) {
    cond_probs.push_back(priors[label]);
    for (int feature = 0; feature < sample.size(); feature++) {
      cond_probs[label] *= pdf[label][feature];
    }
  }

  int max_index = 0;
  double max_prob = 0.0;
  // Return the label of highest probability
  for (int label = 0; label < cond_probs.size(); label++) {
    if(cond_probs[label] > max_prob) {
      max_index = label;
      max_prob = cond_probs[label];
    }
  }

  // Debug prints
  cout << "-------------" << endl;
  // cout << "pdf = " << endl;
  // for (auto i = 0; i < pdf.size(); i++) {
  //   for (auto j = 0; j < pdf[i].size(); j++) {
  //     cout << pdf[i][j] << ' ';
  //   }
  //   cout << endl;
  // }
  cout << "cond_probs = ";
  for (auto i = 0; i < cond_probs.size(); i++) {
    cout << cond_probs[i] << ' ';
  }
  cout << " | max_index = " << max_index << endl;

  return this -> possible_labels[max_index];
}
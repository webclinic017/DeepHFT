
#include <cstdlib>
#include <vector>
#include <cmath>

#include "../lib/linear.hpp"

std::vector<double> LinearRegression::model() { return std::vector<double>({slope, bias}); }
double LinearRegression::predict(double x) { return slope * x + bias; }

double LinearRegression::fit(std::vector<double> &x, std::vector<double> &y) {
    // compute x_mean and y_mean
    double x_mean = 0.00;
    double y_mean = 0.00;
    for(unsigned int i = 0; i < x.size(); i++) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= x.size();
    y_mean /= y.size();
    // compute linear slope and bias term
    double delta_x, delta_y = 0.00;
    for(unsigned int i = 0; i < x.size(); i++) {
        delta_y += x[i] * (y[i] - y_mean);
        delta_x += x[i] * (x[i] - x_mean);
    }
    slope = delta_y / delta_x;
    bias = y_mean - slope * x_mean;
    // compute covariance of x and y
    double cov = 0.00;
    for(unsigned int i = 0; i < x.size(); i++)
        cov += (x[i] - x_mean) * (y[i] - y_mean);
    cov /= x.size();
    // compute standard deviation of x and y
    double x_var, y_var = 0.00;
    for(unsigned int i = 0; i < x.size(); i++) {
        x_var += pow(x[i] - x_mean, 2);
        y_var += pow(y[i] - y_mean, 2);
    }
    double x_stdev = sqrt(x_var / x.size());
    double y_stdev = sqrt(y_var / y.size());
    // pearson correlation coefficient
    r = cov / (x_stdev * y_stdev);

    return r;
}

// --- //

std::vector<double> local_linear_regression(std::vector<double> &dat, unsigned int neighbors) {
    std::vector<double> local_linear;
    // compute local linear regression lines of each k-nearest data points
    for(unsigned int i = 0; i < dat.size(); i++) {
        std::vector<double> euclidean;
        std::vector<unsigned int> indexes;
        for(unsigned int k = 0; k < dat.size(); k++) {
            euclidean.push_back(sqrt(pow(k - i, 2) + pow(dat[k] - dat[i], 2)));
            indexes.push_back(k);
        }
        // sort indexes by lowest to highest euclidean distance
        std::sort(indexes.begin(), indexes.end(), [&](unsigned int i, unsigned int j){return euclidean[i] < euclidean[j];});

        std::vector<double> x = {indexes.begin(), indexes.begin() + neighbors};
        std::vector<double> y;
        for(unsigned int x_k: x) y.push_back(dat[x_k]);

        LinearRegression line;
        line.fit(x, y);

        local_linear.push_back(line.predict(i));

        std::vector<double>().swap(euclidean);
        std::vector<unsigned int>().swap(indexes);
        std::vector<double>().swap(x);
        std::vector<double>().swap(y);
    }

    return local_linear;
}


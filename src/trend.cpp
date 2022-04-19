
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../lib/data.hpp"
#include "../lib/trend.hpp"

void TrendHFT::sample_dataset(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y) {
    //std::system(("./python/download.py historical " + ticker).c_str());
    std::vector<double> dat = read("./temp/dataset");

    for(unsigned int t = 0; t < dat.size() - 14; t++) {
        std::vector<double> x_t = {dat.begin() + t, dat.begin() + t + 10}; // past 10 min
        normalize(x_t);

        // future delta (1 min, 3 min, 5 min)
        double delta1 = relu((dat[t+10] - dat[t+9]) * 100 / dat[t+9]);
        double delta3 = relu((dat[t+12] - dat[t+9]) * 100 / dat[t+9]);
        double delta5 = relu((dat[t+14] - dat[t+9]) * 100 / dat[t+9]);
        std::vector<double> y_t = {delta1, delta3, delta5};

        x.push_back(x_t);
        y.push_back(y_t);

        std::vector<double>().swap(x_t);
        std::vector<double>().swap(y_t);
    }
}

void TrendHFT::build() {
    std::vector<std::vector<double>> x, y;
    sample_dataset(x, y);

    double test = 0.20;
    std::vector<std::vector<double>> train_x = {x.begin(), x.begin() + (int)(x.size() * (1.00 - test))};
    std::vector<std::vector<double>> train_y = {y.begin(), y.begin() + (int)(y.size() * (1.00 - test))};
    std::vector<std::vector<double>> test_x = {x.begin() + (int)(x.size() * (1.00 - test)), x.end()};
    std::vector<std::vector<double>> test_y = {y.begin() + (int)(y.size() * (1.00 - test)), y.end()};

    std::vector<std::vector<double>>().swap(x);
    std::vector<std::vector<double>>().swap(y);

    double alpha = 0.001;
    double decay = 0.001;

    unsigned int epoch       = 10000;
    unsigned int iteration   = 1000;
    unsigned int batch_size  = 100;

    model.train(train_x, train_y, epoch, iteration, batch_size, alpha, decay);

    std::vector<std::vector<double>>().swap(train_x);
    std::vector<std::vector<double>>().swap(train_y);

    // --- //

    double loss = 0.00;
    for(unsigned int i = 0; i < test_x.size(); i++) {
        std::vector<double> yhat = model.predict(test_x[i]);
        loss += mse(test_y[i], yhat);

        std::vector<double>().swap(yhat);
    }
    loss /= test_x.size();

    std::cout << "TEST LOSS = " << loss << "\n";

    std::vector<std::vector<double>>().swap(test_x);
    std::vector<std::vector<double>>().swap(test_y);

    // --- //

    model.save("./models/trend/" + ticker + "/checkpoint");
}


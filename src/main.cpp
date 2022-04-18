
#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "../lib/bar.hpp"
#include "../lib/data.hpp"
#include "../lib/linear.hpp"
#include "../lib/dnn.hpp"

int main(int argc, char *argv[])
{
    std::vector<double> dat = read("./temp/dataset");

    std::vector<std::vector<double>> x, y;
    for(unsigned int t = 0; t < dat.size() - 14; t++) {
        std::vector<double> x_t = {dat.begin() + t, dat.begin() + t + 10};
        normalize(x_t);

        double delta1 = relu((dat[t+10] - dat[t+9]) * 100 / dat[t+9]);
        double delta3 = relu((dat[t+12] - dat[t+9]) * 100 / dat[t+9]);
        double delta5 = relu((dat[t+14] - dat[t+9]) * 100 / dat[t+9]);
        std::vector<double> y_t = {delta1, delta3, delta5};

        x.push_back(x_t);
        y.push_back(y_t);

        std::vector<double>().swap(x_t);
        std::vector<double>().swap(y_t);
    }

    double test = 0.20;
    std::vector<std::vector<double>> train_x = {x.begin(), x.begin() + (int)(x.size() * (1.00 - test))};
    std::vector<std::vector<double>> train_y = {y.begin(), y.begin() + (int)(y.size() * (1.00 - test))};
    std::vector<std::vector<double>> test_x = {x.begin() + (int)(x.size() * (1.00 - test)), x.end()};
    std::vector<std::vector<double>> test_y = {y.begin() + (int)(y.size() * (1.00 - test)), y.end()};

    std::vector<std::vector<double>>().swap(x);
    std::vector<std::vector<double>>().swap(y);

    // --- //

    DNN trendHFT({{10,7},{7,5},{5,3}});

    double alpha = 0.001;
    double decay = 0.01;

    unsigned int epoch       = 10000;
    unsigned int iteration   = 1000;
    unsigned int batch_size  = 100;
    unsigned int batch_start = 0;
    unsigned int batch_end   = batch_size;
    unsigned int batch_num   = 1;

    double loss_t = 0.00;

    for(unsigned int e = 1; e <= epoch; e++) {
        shuffle(train_x, train_y);
        std::cout << "=======================================EPOCH " + std::to_string(e) + "=======================================\n";

        while(batch_start < train_x.size()) {
            std::vector<std::vector<double>> batch_x = {train_x.begin() + batch_start, train_x.begin() + batch_end};
            std::vector<std::vector<double>> batch_y = {train_y.begin() + batch_start, train_y.begin() + batch_end};

            for(unsigned int i = 1; i <= iteration; i++) {
                for(unsigned int k = 0; k < batch_x.size(); k++)
                    trendHFT.fit(batch_x[k], batch_y[k], alpha, train_x.size());

                double loss = 0.00;
                for(unsigned int k = 0; k < batch_x.size(); k++) {
                    std::vector<double> yhat = trendHFT.predict(batch_x[k]);
                    loss += mse(batch_y[k], yhat);

                    std::vector<double>().swap(yhat);
                }
                loss /= batch_size;

                progress_bar(i, iteration, "BATCH " + std::to_string(batch_num) + " [LOSS = " + std::to_string(loss) + "]");
            }

            std::vector<std::vector<double>>().swap(batch_x);
            std::vector<std::vector<double>>().swap(batch_y);

            batch_num++;
            batch_start += batch_size;
            batch_end + batch_size < train_x.size() ? batch_end += batch_size : batch_end = train_x.size();
        }

        std::cout << "\n";

        if(e != 1)
            std::cout << "PREVIOUS EPOCH LOSS = " << loss_t << "\n";

        loss_t = 0.00;
        for(unsigned int k = 0; k < train_x.size(); k++) {
            std::vector<double> yhat = trendHFT.predict(train_x[k]);
            loss_t += mse(train_y[k], yhat);
        }
        loss_t /= train_x.size();

        std::cout << "CURRENT EPOCH LOSS  = " << loss_t << "\n\n";

        batch_start = 0;
        batch_end = batch_size;
        batch_num = 1;

        if(e % (int)(epoch / 10) == 0) alpha *= 1.00 - decay;
    }

    std::vector<std::vector<double>>().swap(train_x);
    std::vector<std::vector<double>>().swap(train_y);

    // --- //

    std::vector<std::vector<double>> yhat;

    loss_t = 0.00;
    for(unsigned int k = 0; k < test_x.size(); k++) {
        std::vector<double> yhat_k = trendHFT.predict(test_x[k]);
        loss_t += mse(test_y[k], yhat_k);

        yhat.push_back(yhat_k);
        std::vector<double>().swap(yhat_k);
    }
    loss_t /= test_x.size();

    std::cout << "TEST LOSS = " << loss_t << "\n";

    std::ofstream out("./temp/results"); // write python code to evaluate test predictions
    if(out.is_open()) {
        for(unsigned int k = 0; k < test_y.size(); k++) {
            for(unsigned int i = 0; i < test_y[k].size(); i++) {
                out << std::to_string(test_y[k][i]);
                if(i != test_y[k].size() - 1)
                    out << " ";
            }
            out << "\n";

            for(unsigned int i = 0; i < yhat[k].size(); i++) {
                out << std::to_string(yhat[k][i]);
                if(i != yhat[k].size() - 1)
                    out << " ";
            }

            if(k != test_y.size() - 1)
                out << "\n";
        }
        out.close();
    }

    std::vector<std::vector<double>>().swap(test_x);
    std::vector<std::vector<double>>().swap(test_y);
    std::vector<std::vector<double>>().swap(yhat);

    return 0;
}


#ifndef __TREND_HPP_
#define __TREND_HPP_

#include <cstdlib>
#include <string>
#include <filesystem>

#include "dnn.hpp"

class TrendHFT
{
private:
    std::string ticker;
    DNN model;
public:
    TrendHFT() {}
    TrendHFT(std::string _ticker): ticker(_ticker) {
        std::filesystem::path model_dir("./models/trend/" + ticker);
        if(!std::filesystem::exists(model_dir))
            std::system(("mkdir ./models/trend/" + ticker).c_str());

        model = DNN({{10,7},{7,5},{5,3}});
        model.load("./models/trend/" + ticker + "/checkpoint");
    }

    void sample_dataset(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y);

    void run();
    void build();
    void update();
};

#endif

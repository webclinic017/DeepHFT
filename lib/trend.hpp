#ifndef __TREND_HPP_
#define __TREND_HPP_

#include <cstdlib>
#include <string>

#include "dnn.hpp"

class TrendHFT
{
private:
    std::string ticker;
    DNN model;
public:
    TrendHFT() {}
    TrendHFT(std::string _ticker): ticker(_ticker) {
        // load existing model
    }

    void run();
    void fit(unsigned int epoch, unsigned int iteration, unsigned int batch_size, double alpha, double decay, double test);
    void update();

    void save();
    void load();
};

#endif

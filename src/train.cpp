
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/data.hpp"
#include "../lib/trend.hpp"

void TrendHFT::fit(unsigned int epoch, unsigned int iteration, unsigned int batch_size, double alpha, double decay, double test) {
    //std::system("./python/download.py historical " + ticker);
    std::vector<double> dat = read("./temp/dataset");
}

// --- //


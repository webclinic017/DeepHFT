
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

#include "../lib/data.hpp"

std::vector<double> read(std::string path) {
    std::vector<double> dat;

    std::ifstream src(path);
    if(src.is_open()) {
        std::string line;
        while(std::getline(src, line))
            dat.push_back(std::stod(line));
        src.close();
    }

    return dat;
}

void write(std::vector<double> &dat, std::string path) {
    std::ofstream out(path);
    if(out.is_open()) {
        for(unsigned int i = 0; i < dat.size(); i++) {
            out << dat[i];
            if(i != dat.size() - 1)
                out << "\n";
        }
        out.close();
    }
}

// --- //

void shuffle(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y) {
    for(unsigned int i = 0; i < x.size(); i++) {
        unsigned int rand_pos = rand() % x.size();
        x[i].swap(x[rand_pos]);
        y[i].swap(y[rand_pos]);
    }
}

// --- //

double mean(std::vector<double> &dat) {
    double sum = 0.00;
    for(double &val: dat)
        sum += val;
    return sum / dat.size();
}

double std_dev(std::vector<double> &dat) {
    double rss = 0.00;
    double u = mean(dat);
    for(double &val: dat)
        rss += pow(val - u, 2);
    return sqrt(rss / dat.size());
}

void normalize(std::vector<double> &dat) {
    double u = mean(dat);
    double sigma = std_dev(dat);
    for(double &val: dat)
        val = (val - u) / sigma;
}

std::vector<double> calculate_residual(std::vector<double> &x, std::vector<double> &y) {
    std::vector<double> residual;
    for(unsigned int i = 0; i < x.size(); i++)
        residual.push_back(x[i] - y[i]);
    return residual;
}


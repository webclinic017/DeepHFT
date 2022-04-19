
#ifndef __DNN_HPP_
#define __DNN_HPP_

#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <string>

double relu(double x);
double relu_prime(double x);

double mse(std::vector<double> &y, std::vector<double> &yhat);

class Node
{
private:
    double sum;
    double b;
    double act;
    double err;
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in_features, std::default_random_engine &generator) {
        init();
        b = 0.00;
        // He-initialization
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        for(unsigned int i = 0; i < in_features; i++)
            w.push_back(standard_normal(generator) * sqrt(2.00 / in_features));
    }

    void init();

    double bias();
    double summation();
    double activation();
    double error();
    std::vector<double> *weights();

    void set_summation(double dot);
    void compute_activation();
    void add_error(double val);
    void update_bias(double delta);
};

class Layer
{
private:
    std::vector<Node> n;
    unsigned int in;
    unsigned int out;
public:
    Layer() {}
    Layer(unsigned int _in, unsigned int _out, std::default_random_engine &generator): in(_in), out(_out) {
        for(unsigned int i = 0; i < out_features(); i++)
            n.push_back(Node(in_features(), generator));
    }

    unsigned int in_features();
    unsigned int out_features();
    std::vector<Node> *nodes();
};

class DNN
{
private:
    std::vector<Layer> layers;
public:
    DNN() {}
    DNN(std::vector<std::vector<unsigned int>> shape) {
        srand(time(NULL));
        std::default_random_engine generator;

        for(unsigned int l = 0; l < shape.size(); l++)
            layers.push_back(Layer(shape[l][0], shape[l][1], generator));
    }

    std::vector<double> predict(std::vector<double> &x);
    void fit(std::vector<double> &x, std::vector<double> &y, double alpha, unsigned int dataset_size);
    void train(std::vector<std::vector<double>> &train_x, std::vector<std::vector<double>> &train_y,
               unsigned int epoch, unsigned int iteration, unsigned int batch_size, double alpha, double decay);

    void save(std::string path);
    void load(std::string path);
};

#endif

// verified: 2022-04-18
#include <cstdlib>
#include <vector>
#include <cmath>

#include "../lib/dnn.hpp"

double relu(double x) { return x > 0.00 ? x : 0.00; }
double relu_prime(double x) { return x > 0.00 ? 1.00 : 0.00; }

double mse(std::vector<double> &y, std::vector<double> &yhat) {
    double loss = 0.00;
    for(unsigned int i = 0; i < y.size(); i++)
        loss += pow(y[i] - yhat[i], 2);
    loss /= y.size();
    return loss;
}

// --- //

void Node::init() {
    sum = 0.00;
    act = 0.00;
    err = 0.00;
}

double Node::summation() { return sum; }
double Node::activation() { return act; }
double Node::error() { return err; }
std::vector<double> *Node::weights() { return &w; }

void Node::set_summation(double val) { sum = val + bias; }
void Node::compute_activation() { act = relu(sum); }
void Node::add_error(double val) { err += val; }
void Node::update_bias(double delta) { bias -= delta; }

// --- //

unsigned int Layer::in_features() { return in; }
unsigned int Layer::out_features() { return out; }
std::vector<Node> *Layer::nodes() { return &n; }

// --- //

std::vector<double> DNN::predict(std::vector<double> &x) {
    std::vector<double> yhat;
    for(unsigned int l = 0; l < layers.size(); l++) {
        std::vector<Node> *nodes = layers[l].nodes();
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            double dot = 0.00;
            std::vector<double> *weights = (*nodes)[n].weights();
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l == 0)
                    dot += x[i] * (*weights)[i];
                else
                    dot += (*layers[l-1].nodes())[i].activation() * (*weights)[i];
            }

            (*nodes)[n].init();
            (*nodes)[n].set_summation(dot);
            (*nodes)[n].compute_activation();

            if(l == layers.size() - 1)
                yhat.push_back((*nodes)[n].activation());
        }
    }

    return yhat;
}

void DNN::fit(std::vector<double> &x, std::vector<double> &y, double alpha, unsigned int dataset_size) {
    std::vector<double> yhat = predict(x);
    // stochastic gradient descent
    for(int l = layers.size() - 1; l >= 0; l--) {
        std::vector<Node> *nodes = layers[l].nodes();

        double partial_gradient = 0.00, gradient = 0.00;
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            if(l == layers.size() - 1)
                partial_gradient = -2.00 / y.size() * (y[n] - yhat[n]) * relu_prime((*nodes)[n].summation());
            else
                partial_gradient = (*nodes)[n].error() * relu_prime((*nodes)[n].summation());

            (*nodes)[n].update_bias(alpha * partial_gradient);

            std::vector<double> *weights = (*nodes)[n].weights();
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l > 0) {
                    gradient = partial_gradient * (*layers[l-1].nodes())[i].activation();
                    (*layers[l-1].nodes())[i].add_error(partial_gradient * (*weights)[i]);
                }
                else
                   gradient = partial_gradient * x[i];

                gradient += 1.00 / dataset_size * (*weights)[i]; // L2 Regularization

                (*weights)[i] -= alpha * gradient;
            }
        }
    }

    std::vector<double>().swap(yhat);
}



#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#include "../lib/dnn.hpp"
#include "../lib/data.hpp"
#include "../lib/bar.hpp"

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

void Node::set_summation(double dot) { sum = dot + bias; }
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

void DNN::train(std::vector<std::vector<double>> &train_x, std::vector<std::vector<double>> &train_y,
                unsigned int epoch, unsigned int iteration, unsigned int batch_size, double alpha, double decay) {
    unsigned int batch_num = 1;
    unsigned int batch_start = 0;
    unsigned int batch_end = batch_size;

    double loss_t = 0.00;

    for(unsigned int e = 1; e <= epoch; e++) {
        shuffle(train_x, train_y);
        std::cout << "=======================================EPOCH " + std::to_string(e) + "=======================================\n";

        while(batch_start < train_x.size()) {
            std::vector<std::vector<double>> batch_x = {train_x.begin() + batch_start, train_x.begin() + batch_end};
            std::vector<std::vector<double>> batch_y = {train_y.begin() + batch_start, train_y.begin() + batch_end};

            for(unsigned int i = 1; i <= iteration; i++) {
                 for(unsigned int k = 0; k < batch_x.size(); k++)
                     fit(batch_x[k], batch_y[k], alpha, train_x.size());

                 double loss = 0.00;
                 for(unsigned int k = 0; k < batch_x.size(); k++) {
                     std::vector<double> yhat = predict(batch_x[k]);
                     loss += mse(batch_y[k], yhat);

                     std::vector<double>().swap(yhat);
                 }
                 loss /= batch_x.size();

                 progress_bar(i, iteration, "BATCH " + std::to_string(batch_num) + " [LOSS = " + std::to_string(loss) + "]");
            }

            std::vector<std::vector<double>>().swap(batch_x);
            std::vector<std::vector<double>>().swap(batch_y);

            batch_num++;
            batch_start += batch_size;
            batch_end + batch_size < train_x.size() ? batch_end += batch_size : batch_end = train_x.size();
        }

        std::cout << "\n";
        if(e != 1) std::cout << "PREVIOUS EPOCH LOSS = " << loss_t << "\n";

        loss_t = 0.00;
        for(unsigned int k = 0; k < train_x.size(); k++) {
            std::vector<double> yhat = predict(train_x[k]);
            loss_t += mse(train_y[k], yhat);

            std::vector<double>().swap(yhat);
        }
        loss_t /= train_x.size();

        std::cout << "CURRENT EPOCH LOSS  = " << loss_t << "\n\n";

        batch_num = 1;
        batch_start = 0;
        batch_end = batch_size;

        if(e % (int)(epoch / 10) == 0) alpha *= 1.00 - decay;
    }
}


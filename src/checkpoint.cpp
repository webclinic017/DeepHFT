
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "../lib/dnn.hpp"

void DNN::save(std::string path) {
    std::ofstream checkpoint(path);
    if(checkpoint.is_open()) {
        for(unsigned int l = 0; l < layers.size(); l++) {
            std::vector<Node> *nodes = layers[l].nodes();
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                std::vector<double> *weights = (*nodes)[n].weights();
                for(unsigned int i = 0; i < layers[l].in_features(); i++)
                    checkpoint << (*weights)[i] << " ";
                checkpoint << (*nodes)[n].bias();
                checkpoint << "\n";
            }
        }
        checkpoint.close();
    }

    for(unsigned int l = 0; l < layers.size(); l++) {
        std::vector<Node> *nodes = layers[l].nodes();
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            std::vector<double> *weights = (*nodes)[n].weights();
            std::vector<double>().swap(*weights);
        }
        std::vector<Node>().swap(*nodes);
    }
    std::vector<Layer>().swap(layers);
}

void DNN::load(std::string path) {
    std::ifstream checkpoint(path);
    if(checkpoint.is_open()) {
        std::string line = "", val = "";
        for(unsigned int l = 0; l < layers.size(); l++) {
            std::vector<Node> *nodes = layers[l].nodes();
            for(unsigned int n = 0; n < layers[l].out_features(); n++) {
                std::vector<double> *weights = (*nodes)[n].weights();
                std::getline(checkpoint, line);

                unsigned int count = 0;
                for(unsigned int i = 0; i < line.length(); i++) {
                    if(line[i] != ' ' || i == line.length() - 1) {
                        val += line[i];
                        if(i == line.length() - 1)
                            (*nodes)[n].update_bias(-1.00 * std::stod(val));
                    }
                    else {
                        (*weights)[count] = std::stod(val);
                        std::cout << (*weights)[count] << "\n";
                        count++;

                        val = "";
                    }
                }
                val = "";
            }
        }
        checkpoint.close();
    }
}


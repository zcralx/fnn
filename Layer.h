//
// Created by rui zhan on 2024/5/28.
//

#ifndef NETWORK_PLUS_LAYER_H
#define NETWORK_PLUS_LAYER_H
#include "iostream"
#include "vector"
#include "Neuron.h"

class Layer {
private:
    std::vector<Neuron> l_neurons;
    std::vector<double> l_outputs;
public:
    Layer(size_t neuron_count);
    Layer(size_t neuron_count, size_t input_count);
    std::vector<Neuron>& getNeurons();
    std::vector<double>& getOutputs();
    Neuron& getNeuron(size_t index);
    void forward_pass(const double& input);
    void forward_pass(const std::vector<double>& inputs);
};


#endif //NETWORK_PLUS_LAYER_H

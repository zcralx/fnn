//
// Created by rui zhan on 2024/5/28.
//

#include "Layer.h"

std::vector<Neuron> &Layer::getNeurons() {
    return l_neurons;
}
std::vector<double> &Layer::getOutputs() {
    return l_outputs;
}
Layer::Layer(size_t neuron_count) {
    l_neurons.reserve(neuron_count);
    for (size_t i = 0; i < neuron_count; i++) {
        l_neurons.push_back(Neuron());
    }
}
Layer::Layer(size_t neuron_count, size_t input_count) {
    /*
     * neuron_count:该层神经元个数
     * input_count:该层输入个数
     * */
    l_neurons.reserve(neuron_count);
    for (size_t i = 0; i < neuron_count; i++) {
        l_neurons.push_back(Neuron(input_count));
    }
    l_outputs.reserve(neuron_count);
}

Neuron &Layer::getNeuron(size_t index) {
    if(index >= l_neurons.size()){
        throw std::out_of_range("index out of range");
    }
    return l_neurons[index];
}

void Layer::forward_pass(const double &input) {
    for(Neuron& neuron : l_neurons){
        neuron.compute_outputVal(input);
        l_outputs.push_back(neuron.getOutputVal());
    }
}

void Layer::forward_pass(const std::vector<double> &inputs) {
    /*
     * 获取该层所有神经元的输出结果
     * */
    l_outputs.clear();
    for(Neuron& neuron : l_neurons){
        neuron.compute_outputVal(inputs);
        l_outputs.push_back(neuron.getOutputVal());
    }
}

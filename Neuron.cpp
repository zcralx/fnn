//
// Created by rui zhan on 2024/5/28.
//

#include "Neuron.h"
#include "Utils.h"
Neuron::Neuron() {
    n_bias = Utils::random_generator();
    n_weight = Utils::random_generator();
}
Neuron::Neuron(size_t input_size) {
    n_bias = Utils::random_generator();
    n_weights.resize(input_size);
    for (auto& w : n_weights) {
        w = Utils::random_generator();
    }
}
double Neuron::getBias() const {
    return n_bias;
}
double Neuron::getWeight() const {
    return n_weight;
}
const std::vector<double>& Neuron::getWeights() const {
    return n_weights;
}

std::vector<double> &Neuron::getNoConstWeights() {
    return n_weights;
}
double Neuron::getOutputVal() const {
    return n_outputVal;
}
void Neuron::setBias(double bias) {
    n_bias = bias;
}
void Neuron::setWeight(double weight) {
    n_weight = weight;
}
void Neuron::setWeights(const std::vector<double> &weights) {
    n_weights = weights;
}

void Neuron::compute_outputVal(double input_val) {
    n_outputVal = activation_function(n_bias + n_weight * input_val);
}

void Neuron::compute_outputVal(const std::vector<double> &input_vals) {
    if(input_vals.size() != n_weights.size()){
        throw std::runtime_error("神经元的输入个数与权重个数不匹配");
    }
    n_outputVal = activation_function(Utils::addBy2Vector(input_vals, n_weights) + n_bias);
}
double Neuron::activation_function(double x){
    return Utils::Sigmod(x);
}
void Neuron::compute_loss(double target_val) {
    n_loss = pow((target_val - n_outputVal),2);
}
double Neuron::getLoss() const {
    return n_loss;
}

void Neuron::setLoss(double new_loss) {
    n_loss = new_loss;
}
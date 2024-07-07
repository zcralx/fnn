//
// Created by rui zhan on 2024/5/28.
//

#ifndef NETWORK_PLUS_NEURON_H
#define NETWORK_PLUS_NEURON_H
#include "iostream"
#include "vector"

class Neuron {
private:
    double n_outputVal;
    double n_bias;
    double n_weight;
    std::vector<double> n_weights;
    double activation_function(double x);
    double n_loss;
public:
    Neuron();
    Neuron(size_t input_size);
    double getOutputVal() const;
    double getBias() const;
    double getWeight() const;
    const std::vector<double>& getWeights() const;
    std::vector<double>& getNoConstWeights();
    void setBias(double);
    void setWeight(double);
    void setWeights(const std::vector<double>&);
    void compute_outputVal(const std::vector<double>&);
    void compute_outputVal(double);
    void compute_loss(double);
    double getLoss() const;
    void setLoss(double);
};


#endif //NETWORK_PLUS_NEURON_H

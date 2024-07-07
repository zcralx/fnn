//
// Created by rui zhan on 2024/5/28.
//

#ifndef NETWORK_PLUS_FEEDFORWARDNEURALNETWORK_H
#define NETWORK_PLUS_FEEDFORWARDNEURALNETWORK_H
#include "Layer.h"

class FeedforwardNeuralNetwork {
private:
    std::vector<Layer> f_layers;
    std::vector<double> f_outputs;
    std::vector<double> f_inputs;
    std::vector<double> last_layer_loss;
    double learning_rate;
    double total_loss;
public:
    FeedforwardNeuralNetwork(std::vector<size_t> layers_sizes);
    FeedforwardNeuralNetwork(std::vector<size_t> layers_sizes, size_t inputs_size);
    void forward(const std::vector<double>& inputs);
    const std::vector<double>& getOutputs() const;
    const double getLearningRate() const;
    void setLearningRate(double);
    void compute_loss(const std::vector<double>& predictions);
    double getLoss() const;
    const std::vector<double>& getLastLayerLoss() const;
    void backpropagation();
    std::vector<Layer> &getF_layers();
};


#endif //NETWORK_PLUS_FEEDFORWARDNEURALNETWORK_H

//
// Created by rui zhan on 2024/5/28.
//

#include "FeedforwardNeuralNetwork.h"
#include "Utils.h"

FeedforwardNeuralNetwork::FeedforwardNeuralNetwork(std::vector<size_t> layers_sizes) {
    f_outputs.reserve(layers_sizes.back());
    f_inputs.reserve(1);
    last_layer_loss.reserve(layers_sizes.back());
    f_layers.reserve(layers_sizes.size());
    f_layers.push_back(Layer(layers_sizes[0]));
    for(size_t i = 1; i < layers_sizes.size(); i++){
        f_layers.push_back(Layer(layers_sizes[i], layers_sizes[i-1]));
    }
}

FeedforwardNeuralNetwork::FeedforwardNeuralNetwork(std::vector<size_t> layers_sizes, size_t inputs_size) {
    f_outputs.reserve(layers_sizes.back());
    f_inputs.reserve(inputs_size);
    last_layer_loss.reserve(layers_sizes.back());
    f_layers.reserve(layers_sizes.size());
    f_layers.push_back(Layer(layers_sizes[0], inputs_size));
    for(size_t i = 1; i < layers_sizes.size(); i++){
        f_layers.push_back(Layer(layers_sizes[i], layers_sizes[i-1]));
    }
}

const std::vector<double> &FeedforwardNeuralNetwork::getOutputs() const {
    return f_outputs;
}

const double FeedforwardNeuralNetwork::getLearningRate() const {
    return learning_rate;
}

void FeedforwardNeuralNetwork::setLearningRate(double new_learning_rate) {
    learning_rate = new_learning_rate;
}

void FeedforwardNeuralNetwork::compute_loss(const std::vector<double>& predictions) {
    /*
     * 采用均方误差计算损失
     * last_layer_loss: 存放输出层的损失
     * */
    total_loss = 0.0;
    last_layer_loss.clear();
    if(predictions.size() != f_outputs.size()){
        throw std::invalid_argument("predictions size must be equal to outputs size");
    }
    for(size_t i = 0; i < f_outputs.size(); i++){
        f_layers[f_layers.size()-1].getNeuron(i).compute_loss(predictions[i]);
        last_layer_loss.push_back(f_layers[f_layers.size()-1].getNeuron(i).getLoss() / f_outputs.size());
    }
    for(auto& loss : last_layer_loss){
        total_loss += loss;
    }
}

double FeedforwardNeuralNetwork::getLoss() const {
    return total_loss;
}

const std::vector<double> &FeedforwardNeuralNetwork::getLastLayerLoss() const {
    return last_layer_loss;
}

void FeedforwardNeuralNetwork::forward(const std::vector<double> &inputs) {
    f_inputs = inputs;
    f_layers[0].forward_pass(inputs);
    for(size_t i = 1; i < f_layers.size(); i++){
        f_layers[i].forward_pass(f_layers[i-1].getOutputs());
    }
    f_outputs = f_layers.back().getOutputs();
}

void FeedforwardNeuralNetwork::backpropagation() {
    /*
     * 根据total_loss计算每个神经元的误差
     * 包括输出层在内至少应该是2层
     * */
    for(int i = f_layers.size() - 2; i >= 0; i--){
        for(size_t k = 0;k < f_layers[i].getNeurons().size(); k++){
            for(size_t j = 0;j < f_layers[i+1].getNeurons().size(); j++){
                f_layers[i].getNeuron(k).setLoss(
                        f_layers[i+1].getNeuron(j)
                        .getWeights()[k] * f_layers[i+1]
                        .getNeuron(j).getLoss()
                );
            }
        }
    }
    /*
     * 更新第0层的权重需要得到输入的数据
     * */
    for(size_t j = 0;j < f_layers[0].getNeurons().size();j++){
        double temp = f_layers[0].getNeuron(j).getLoss() * learning_rate * (1 - f_layers[0].getNeuron(j).getOutputVal()) * f_layers[0].getNeuron(j).getOutputVal();
        for(size_t k = 0;k < f_layers[0].getNeuron(0).getWeights().size(); k++){
            f_layers[0].getNeuron(j).getNoConstWeights()[k] += temp * f_inputs[k];
        }
    }
    /*
     * 更新其他层的权重
     * */
    for(size_t i = 1;i < f_layers.size(); i++){
        for(size_t j = 0;j < f_layers[i].getNeurons().size(); j++){
            double temp = f_layers[i].getNeuron(j).getLoss() * learning_rate * (1 - f_layers[i].getNeuron(j).getOutputVal()) * f_layers[i].getNeuron(j).getOutputVal();
            for(size_t k = 0;k < f_layers[i].getNeuron(0).getWeights().size(); k++){
                f_layers[i].getNeuron(j).getNoConstWeights()[k] += temp * f_layers[i-1].getOutputs()[k];
            }
        }
    }
}
std::vector<Layer> &FeedforwardNeuralNetwork::getF_layers() {
    return f_layers;
}
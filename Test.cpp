//
// Created by rui zhan on 2024/5/28.
//

#include "Test.h"
#include "Utils.h"
#include "fstream"

void Test::test_layer() {
    Layer layer = Layer(3, 2);
    layer.forward_pass({1,2});
    for(Neuron &neuron: layer.getNeurons()){
        std::cout << "neuron: " << std::endl;
        Utils::print(neuron.getOutputVal());
        Utils::printVector(neuron.getWeights());
    }
    Utils::printVector(layer.getOutputs());
}
void Test::test_network() {
    FeedforwardNeuralNetwork network = FeedforwardNeuralNetwork({3,2},2);
    network.forward({1,2});
    network.compute_loss({1,2});
    Utils::printVector(network.getOutputs());
    Utils::print(network.getLoss());
}
void Test::test_back(){
    FeedforwardNeuralNetwork network = FeedforwardNeuralNetwork({3,2},2);
    network.setLearningRate(0.5);
    network.forward({1,2});
    network.compute_loss({1,2});
    Utils::printVector(network.getOutputs());
    Utils::print(network.getLoss());
    printf("-------------\n");
    for(Layer &layer: network.getF_layers()){
        for(Neuron &neuron: layer.getNeurons()){
            std::cout << "*******" << std::endl;
            Utils::printVector(neuron.getWeights());
        }
    }
    printf("-------------\n");
    network.backpropagation();
    network.forward({1,2});
    network.compute_loss({1,2});
    Utils::printVector(network.getOutputs());
    Utils::print(network.getLoss());
    printf("-------------\n");
    for(Layer &layer: network.getF_layers()){
        for(Neuron &neuron: layer.getNeurons()){
            std::cout << "*******" << std::endl;
            Utils::printVector(neuron.getWeights());
        }
    }
}
void Test::test_plus(){
    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return;
    }
    FeedforwardNeuralNetwork network = FeedforwardNeuralNetwork({5, 3, 2},2);
    network.setLearningRate(0.5);
    network.forward({1,2});
    network.compute_loss({1, 2});
    //Utils::printVector(network.getOutputs());
    //Utils::print(network.getLoss());
    outputFile << network.getLoss() << "\n";
    //printf("-------------\n");
    for(int i = 0; i < 500;i++){
        //printf("第%d次\n",i+1);
        network.backpropagation();
        network.forward({1,2});
        network.compute_loss({1, 2});
        //Utils::printVector(network.getOutputs());
        //Utils::print(network.getLoss());
        //printf("-------------\n");
        outputFile << network.getLoss() << "\n";
    }
    outputFile.close();
    std::cout << "Output written to output.txt" << std::endl;
}
void Test::test_neruon() {
    Neuron neuron = Neuron(3);
    Utils::printVector(neuron.getWeights());
    std::cout << neuron.getBias() << std::endl;
    neuron.compute_outputVal({1,2,3});
    Utils::print(neuron.getOutputVal());
}
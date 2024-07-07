//
// Created by rui zhan on 2024/5/28.
//

#include "Utils.h"
#include "random"
double Utils::random_generator(double mu) {
    /*
     * 生成一个以mu为均值，标准差为1.0的正态分布随机数
     * */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(mu, 1.0);
    return normal_dist(gen);
}
void Utils::print(double a) {
    printf("value: %f\n", a);
}
void Utils::printVector(const std::vector<double>& vector) {
    printf("values: ");
    for (auto i : vector) {
        printf("%f ", i);
    }
    printf("\n");
}
void Utils::printVector(const std::vector<std::vector<double>>& vector){
    printf("values: \n");
    for(size_t i = 0;i < vector.size(); ++i){
        for(auto& ve : vector[i]){
            printf("%f ", ve);
        }
        printf("\n");
    }
}

double Utils::Sigmod(double x) {
    return 1 / (1 + exp(-x));
}
double Utils::meanSquaredError(const std::vector<double> predictions, const std::vector<double> true_outputs) {
    /*
     * 均方误差(Mean Squared Error, MSE)
     * predictions:预测值, true_outputs:真实值
     * */
    double loss = 0.0;
    for(size_t i = 0;i < predictions.size(); ++i){
        loss += pow(predictions[i] - true_outputs[i], 2);
    }
    return loss / predictions.size();
}
double Utils::binaryCrossEntropy(const std::vector<double> predictions, const std::vector<double> targets) {
    /*
     * 二元交叉熵损失函数
     * predictions:预测值, targets:真实值
     * */
    double loss = 0.0;
    for(size_t i = 0;i < predictions.size(); ++i){
        if(predictions[i] > 0 && predictions[i] < 1){
            loss -= targets[i] * log(predictions[i]) + (1 - targets[i]) * log(1 - predictions[i]);
        } else {
            float epsilon = 1e-7;
            loss -= targets[i] * log(predictions[i] + epsilon) + (1 - targets[i]) * log(1 - predictions[i] + epsilon);
        }
    }
    return loss / predictions.size();
}

double Utils::addBy2Vector(const std::vector<double> &a, const std::vector<double> &b) {
    if(a.size() != b.size()){
        throw std::invalid_argument("The size of the two vectors must be the same.");
    }
    double result = 0.0;
    for(size_t i = 0;i < a.size(); ++i){
        result += a[i] * b[i];
    }
    return result;
}


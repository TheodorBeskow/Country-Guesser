#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include "node.hpp"
#include "weight.hpp"

typedef long double ld;

class ActivationFunctions{
public:
    ld sigmoid(ld x){
        return 1 / (1 + exp(-x));
    }
    ld sigmoid_derivative(ld x){
        return x*(x-1);
    }
    
};


class NeuralNetwork{
public:
    NeuralNetwork(std::vector<int> layers){
        assert(layers.size());
        for(int i = 0; i<layers.size(); i++){
            nodes.push_back(std::vector<node*>(layers[i], new node()));
        }
        weights.resize(((int)layers.size())-1);
        for(int i = 0; i<weights.size(); i++){
            weights[i].assign(layers[i], std::vector<weight*>(layers[i+1], new weight()));
        }
    }
    void backwardPropagation(){
        
    }
    std::vector<ld> forwardPropagation(std::vector<ld> inNodes){
        for(int i = 0; i<inNodes.size(); i++) nodes[0][i]->value = inNodes[i];
        
    }
    ~NeuralNetwork(){}

private:
    std::vector<std::vector<node*>> nodes;
    std::vector<std::vector<std::vector<weight*>>> weights;



};
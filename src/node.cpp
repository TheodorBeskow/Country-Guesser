#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

typedef long double ld;
// std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());


class node{
public:
    ld value;
    node(){
        resetValue();
    }
    ~node(){}

    void resetValue(){
        value = 0;
    }
};


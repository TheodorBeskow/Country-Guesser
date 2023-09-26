#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

typedef long double ld;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());


class weight{
public:
    ld value;
    weight(){
        value = ((ld)(rng()%((int)2e9)-1e9))/1e9; // random value [-1, 1]
        std::cout << value << std::endl;
    }
    ~weight(){}
private:
};


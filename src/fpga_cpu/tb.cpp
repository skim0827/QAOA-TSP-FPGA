#include "qaoa.hpp"
# include <iostream>
#include <bitset>
#include <chrono>



int main(){ 
    qfix d[3][3] = { {0.0, 10.0, 4.7}, {10.0, 0.0, 11.0}, {4.7, 11.0, 0.0} };
    qfix gamma[1] = { 0.8 };
    qfix beta[1]  = { 1.5 };
    uint32_t best_state = 0; 
    auto start = std::chrono::high_resolution_clock::now();
    qfix expectation;   // real storage
    qaoa_kernel(d, gamma, beta, true, &best_state, &expectation); 

    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << "Expectation value = " << (double)expectation << std::endl;
    std::cout << "Best state        = " << std::bitset<9>(best_state) << std::endl;
    std::cout << "CPU Time          = " << ms << " ms\n";
}
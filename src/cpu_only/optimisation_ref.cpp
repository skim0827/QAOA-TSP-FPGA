#include "qaoa_ref.hpp"
# include <iostream>
#include <cmath>
#include <limits>
#include <bitset>
#include <nlopt.hpp>
#include <vector>
#include <chrono>
// screen /dev/ttyUSB0 115200
// picocom /dev/ttyUSB0 -b 115200
// ls /dev/ttyUSB*
// Running purely on CPU with double
/*
Measure:
time per iteration
total optimisation time
final expectation value
*/


static double d[3][3] = {
    {0.0, 10.0, 4.7},
    {10.0, 0.0, 11.0},
    {4.7, 11.0, 0.0}
};

template<int N_CITY>
uint32_t brute_force(const double d[N_CITY][N_CITY], double &min_energy_out) {
    const int DIM = Config<N_CITY>::DIM; // number of bitstrings = 2^(N_CITY*N_CITY)
    double min_energy = 1e9;
    uint32_t best_state = 0;

    for (uint32_t s = 0; s < DIM; ++s) {
        double energy = costHamiltonian<N_CITY>(s, d);
        if (energy < min_energy) {
            min_energy = energy;
            best_state = s;
        }
    }
    min_energy_out = min_energy;
    return best_state;
}

// =========================================================
// NLopt objective for CPU ONLY
// =========================================================
double objective_function(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    (void)grad; (void)data;
    double gamma[1] = { x[0] };
    double beta[1]  = { x[1] };
    uint32_t best;
    double expectation;

    qaoa_kernel(d, gamma, beta, false, &best, &expectation);
    
    return expectation;
}

// =========================================================
// main 
// =========================================================

int main() {

    // --- Set up NLopt (Nelder–Mead) ---
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 2); // parameters to optimise
    opt.set_lower_bounds({0.0, 0.0});
    opt.set_upper_bounds({M_PI, M_PI});
    opt.set_min_objective(objective_function, nullptr);
    opt.set_xtol_rel(1e-4);
    
    std::vector<double> x = {0.5, 0.5}; // γ1 β1 γ2 β2

    // measure cpu time 
    auto t0 = std::chrono::high_resolution_clock::now();
    double min_expectation ;

    nlopt::result result = opt.optimize(x, min_expectation);
    auto t1 = std::chrono::high_resolution_clock::now();

    qfix gamma[3] = { x[0] };
    qfix beta[3]  = { x[1] };
    qfix expectation_qfix;
    uint32_t best_state_qaoa;
    qaoa_kernel(d, gamma, beta, true, &best_state_qaoa, &expectation_qfix);

    double E_qaoa_bitstring = costHamiltonian<3>(best_state_qaoa, d);
    std::cout << "Energy of QAOA bitstring = "
          << costHamiltonian<3>(best_state_qaoa, d) << std::endl;

    std::cout << "============== Optimisation finished ==================\n";
    std::cout << "gamma = " << x[0] << "\n";
    std::cout << "beta  = " << x[1] << "\n";
    std::cout << "expectation = " << min_expectation << "\n"; 
    std::cout << "QAOA best bitstring = "
          << std::bitset<9>(best_state_qaoa) << std::endl;
    std::cout << "CPU time(ms)="
              << std::chrono::duration<double,std::milli>(t1-t0).count()
              << "\n";
    std::cout << "============== Brut Force ==================\n";
    double min_energy;
    uint32_t best_state_brute = brute_force<3>(d, min_energy);

    std::cout << "Brute-force best bitstring = "
            << std::bitset<9>(best_state_brute) << std::endl;
    std::cout << "Brute-force energy = " << min_energy << std::endl;


    return 0;
}


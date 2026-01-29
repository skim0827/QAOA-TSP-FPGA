#ifndef QAOA_HPP
#define QAOA_HPP
#pragma once


#ifdef __VITIS_HLS__
  #include <hls_math.h>
  #include "ap_fixed.h"
  typedef ap_fixed<32,12> qfix;
  #define HLS_PIPELINE
  #define HLS_UNROLL
  #define HLS_DATAFLOW
#else
  #include <cmath>
  #define HLS_PIPELINE
  #define HLS_UNROLL
  #define HLS_DATAFLOW
  typedef double qfix;
#endif

#include <cstdint>
// Define PI since M_PI is not standard C++
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =======================================================
// 1. Configuration 
// =======================================================

/**
 * Configuration structure for N_CITIES.
 */
template<int N_CITY>
struct Config {
    static const int nCities = N_CITY;
    static const int nQubits = N_CITY * N_CITY;
    // DIM = 2^(N_QUBITS)
    static const int DIM = 1 << (N_CITY * N_CITY);
};
typedef Config<3> CFG; // Default 3 cities

// =======================================================
// 2. Complex Number Class
// =======================================================

struct ComplexQ {
    qfix re;
    qfix im;

    ComplexQ(): re(0.0), im(0.0) {}
    ComplexQ(qfix _r, qfix _i): re(_r), im(_i) {}

    // Operator overloads (inline definitions kept here for efficiency)
    inline ComplexQ operator*(const ComplexQ& o) const {
        return ComplexQ(re*o.re - im*o.im, re*o.im + im*o.re);
    }
    inline ComplexQ operator+(const ComplexQ& o) const {
        return ComplexQ(re + o.re, im + o.im);
    }
    inline ComplexQ operator-(const ComplexQ& o) const {
        return ComplexQ(re - o.re, im - o.im);
    }
    inline ComplexQ& operator*=(const ComplexQ& o) {
        qfix temp_re = re*o.re - im*o.im;
        qfix temp_im = re*o.im + im*o.re;
        re = temp_re;
        return *this;
    }
};

// =======================================================
// 3. Function Declarations
// =======================================================

/**
 * Calculates the Z-eigenvalue for a state 's' at qubit 'q'.
 * Corresponds to (-1)^(bit q of s).
 * Returns +1 if the bit is 0 (|0> state)
 * Returns -1 if the bit is 1 (|1> state)
 */
inline int Z_eigenvalue_uint(uint32_t s, int q) {
    // Check the q-th bit of s
    if ((s >> q) & 1u) {
        return -1; // Qubit is |1>, Z eigenvalue is -1
    } else {
        return 1;  // Qubit is |0>, Z eigenvalue is +1
    }
}
/**
 * Calculates the classical cost (eigenvalue) H_s for a single bitstring state 's'.
 */
template<int N_CITY>
qfix costHamiltonian(uint32_t s, const qfix d[N_CITY][N_CITY]);

/**
 * Initializes the state vector to the equal superposition state |+>^n.
 */
template<int N_CITY>
int build_feasible_superposition(ComplexQ state[Config<N_CITY>::DIM]);

template<int N_CITY>
bool is_valid_onehot(uint32_t s);

/**
 * Applies the Cost Unitary U_C(gamma) = exp(-i * gamma * H_C) to the state vector.
 */
template<int N_CITY>
void applyCost_hls(ComplexQ state[Config<N_CITY>::DIM], 
                    const qfix d[N_CITY][N_CITY], 
                    qfix gamma);

/**
 * Applies the Mixer Unitary U_B(beta) = \prod_q R_X(2*beta)_q to the state vector.
 */
template<int N_CITY>
void applyMixer_hls(ComplexQ state[Config<N_CITY>::DIM], 
                    qfix beta);


template<int N_CITY>
qfix expectation_cost(ComplexQ state[Config<N_CITY>::DIM], 
                        const qfix d[N_CITY][N_CITY], 
                        uint32_t *best_state);
/**
 * Top-level function for a single QAOA layer (U_B * U_C). (HLS Entry Point)
 */
template<int N_CITY, int P>
void qaoaStep_hls(ComplexQ state[Config<N_CITY>::DIM],
                  const qfix d[N_CITY][N_CITY],
                  const qfix gamma[P],
                  const qfix beta[P]);

extern "C"
void qaoa_kernel(const qfix d[3][3],
                   const qfix gamma[1],
                   const qfix beta[1],
                   bool get_best_state,
                   uint32_t *best_state,
                   qfix *expectation);

#endif // QAOA_H

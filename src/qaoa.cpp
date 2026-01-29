#include "qaoa.hpp"
#include <iostream> // Included for HLS debugging/testing context (though not used in final HLS)

template<int N_CITY>
qfix costHamiltonian(uint32_t s, const qfix d[N_CITY][N_CITY]){
    qfix H = 0.0;
    qfix P  = 30;

    for (int i = 0; i < N_CITY; ++i){
    #ifdef __VITIS_HLS__
    #pragma HLS PIPELINE II=1
    #endif 
        for (int j = 0; j < N_CITY; ++j){
    #pragma HLS UNROLL
            for (int t = 0; t < N_CITY; ++t){
                int next_t = (t + 1) % N_CITY;
                int Z_i_t = Z_eigenvalue_uint(s, (t*N_CITY) + i);
                int Z_j_t_1 = Z_eigenvalue_uint(s, (next_t*N_CITY) + j);
                H += d[i][j] * qfix(0.25) * (qfix(1) - Z_i_t) * (qfix(1) - Z_j_t_1);

            }
        }

    }
    // std::cout << "distance contribution = " << H << std::endl;

    for (int i=0; i < N_CITY; i++) {
        qfix X_sum = 0.0;
        for (int t=0; t < N_CITY; t++) {
            X_sum += qfix(0.5) * (qfix(1) - Z_eigenvalue_uint(s, t*N_CITY + i));
        }
        H += P * (X_sum - qfix(1)) * (X_sum - qfix(1)); 
    }

    for (int t=0; t < N_CITY ; t++) {
        qfix X_sum = 0.0;
        for (int i=0; i < N_CITY; i++) {
            X_sum +=qfix(0.5)* (qfix(1) - Z_eigenvalue_uint(s, t*N_CITY + i));
        }
        H += P * (X_sum - qfix(1) ) * (X_sum - qfix(1) );
    }
    
    return H;
}
template<int N_CITY>
int build_feasible_superposition(ComplexQ state[Config<N_CITY>::DIM]) {
    #ifdef __VITIS_HLS__
    #pragma HLS INLINE off
    #endif 
    const int DIM = Config<N_CITY>::DIM;
    int count = 0;
    for (int s = 0; s < DIM; ++s) {
        if (is_valid_onehot<N_CITY>(s)) ++count;
    }
    #ifdef __VITIS_HLS__
    const qfix norm = qfix(1) / hls::sqrt((qfix)count);
    #else
    const qfix norm = qfix(1) / std::sqrt((qfix)count);
    #endif
    for (int s = 0; s < DIM; ++s) {
        state[s] = is_valid_onehot<N_CITY>(s) ? ComplexQ(norm, 0.0) : ComplexQ(0.0, 0.0);
    }
    return count;
}

template<int N_CITY>
bool is_valid_onehot(uint32_t s) {
    for (int t = 0; t < N_CITY; ++t) {
        int ones = 0;
        for (int i = 0; i < N_CITY; ++i) {
            ones += ((s >> (t * N_CITY + i)) & 1u);
        }
        if (ones != 1) return false;
    }
    return true;
}

template<int N_CITY>
void applyCost_hls(ComplexQ state[Config<N_CITY>::DIM], 
                    const qfix d[N_CITY][N_CITY], 
                    qfix gamma) {
    #ifdef __VITIS_HLS__
    #pragma HLS INLINE off
    #endif 
    const int DIM = Config<N_CITY>::DIM;
    for (uint32_t s = 0; s < (uint32_t)DIM; s++) {
        qfix Hs = costHamiltonian<N_CITY>(s, d);
        qfix ang = gamma * Hs;
        #ifdef __VITIS_HLS__
        state[s] *= ComplexQ(hls::cos(ang), -hls::sin(ang));
        #else
        state[s] *= ComplexQ(std::cos(ang), -std::sin(ang));
        #endif

    }
}

template<int N_CITY>
void applyMixer_hls(ComplexQ state[Config<N_CITY>::DIM], qfix beta) {
    #pragma HLS INLINE off
    #ifdef __VITIS_HLS__
    const qfix c = hls::cos(qfix(2) * beta);
    const qfix s = hls::sin(qfix(2) * beta);
    #else
    const qfix c = std::cos(2.0 * beta);
    const qfix s = std::sin(2.0 * beta);
    #endif
    const uint32_t slice_mask = (1u << N_CITY) - 1u;

    // Two buffers to ping-pong between
    ComplexQ bufA[Config<N_CITY>::DIM];
    ComplexQ bufB[Config<N_CITY>::DIM];

    // Start in bufA
    
    for (uint32_t k = 0; k < (uint32_t)Config<N_CITY>::DIM; ++k) bufA[k] = state[k];


    ComplexQ* state_cur  = bufA;
    ComplexQ* state_next = bufB;
    #ifdef __VITIS_HLS__
    #pragma HLS RESOURCE variable=state core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=bufA core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=bufB core=RAM_2P_BRAM
    #endif 

    for (int t = 0; t < N_CITY; ++t) {
        const int base = t * N_CITY;

        // For each pair (i,j), apply a 2×2 rotation and then swap buffers
        for (int i = 0; i < N_CITY; ++i) {
            for (int j = i + 1; j < N_CITY; ++j) {
                // Start from current state; copy unchanged amplitudes forward
                for (uint32_t k = 0; k < (uint32_t)Config<N_CITY>::DIM; ++k) state_next[k] = state_cur[k];

                const uint32_t bit_i = 1u << (t * N_CITY + i);
                const uint32_t bit_j = 1u << (t * N_CITY + j);
                const uint32_t mask  = bit_i | bit_j;

                for (uint32_t s_idx = 0; s_idx < (uint32_t)Config<N_CITY>::DIM; ++s_idx) {
                    const uint32_t slice = (s_idx >> base) & slice_mask;
                #pragma HLS PIPELINE II=1
                    // Only process when the slice has |i> active; we’ll write both partner entries.
                    if (slice == (1u << i)) {
                        const uint32_t s_flip = s_idx ^ mask; // toggles i and j
                        const ComplexQ a = state_cur[s_idx];
                        const ComplexQ b = state_cur[s_flip];

                        // [ a' ] = [  c   -i s ] [ a ]
                        // [ b' ]   [ -i s   c  ] [ b ]
                        state_next[s_idx]  = ComplexQ(c, 0.0) * a - ComplexQ(0.0, s) * b;
                        state_next[s_flip] = ComplexQ(0.0,-s) * a + ComplexQ(c, 0.0) * b;
                    }
                }

                // Compose this rotation: swap roles so the next rotation uses updated state
                ComplexQ* tmp = state_cur;
                state_cur = state_next;
                state_next = tmp;

            }
        }
    }

    // Write back the fully composed result
    for (uint32_t k = 0; k < (uint32_t)Config<N_CITY>::DIM; ++k) state[k] = state_cur[k];
}

template<int N_CITY>
qfix expectation_cost(ComplexQ state[Config<N_CITY>::DIM], const qfix d[N_CITY][N_CITY], uint32_t *best_state){
    #ifdef __VITIS_HLS__
    #pragma HLS INLINE off
    #endif 
    qfix result = 0.0; 
    qfix max_prob = -1.0;
    uint32_t argmax = 0; 
    for (int s = 0; s < Config<N_CITY>::DIM; s++){
    #ifdef __VITIS_HLS__
    #pragma HLS PIPELINE II=1
    #endif 
        qfix prob = state[s].re*state[s].re + state[s].im*state[s].im;
        qfix Hs = costHamiltonian<N_CITY>(s, d);
        result += prob * Hs;

        if (prob >  max_prob) {
            max_prob = prob ;
            argmax = (uint32_t) s;
        }

    }

    *best_state = argmax; 
    return result ; 
}

template<int N_CITY, int P>
void qaoaStep_hls(ComplexQ state[Config<N_CITY>::DIM], const qfix d[N_CITY][N_CITY], const qfix gamma[P], const qfix beta[P]){
    build_feasible_superposition<N_CITY>(state);

    // apply P layers 
    for (int p = 0; p < P; ++p) {
        applyCost_hls<N_CITY>(state, d, gamma[p]);
        applyMixer_hls<N_CITY>(state, beta[p]);        
    }
}

extern "C"
void qaoa_kernel(const qfix d[3][3],
                   const qfix gamma[1],
                   const qfix beta[1],
                   bool get_best_state,
                   uint32_t *best_state,
                   qfix *expectation) {
#pragma HLS INTERFACE s_axilite port=return         bundle=control
#pragma HLS INTERFACE s_axilite port=d              bundle=control
#pragma HLS INTERFACE s_axilite port=gamma          bundle=control
#pragma HLS INTERFACE s_axilite port=beta           bundle=control
#pragma HLS INTERFACE s_axilite port=get_best_state bundle=control
#pragma HLS INTERFACE s_axilite port=best_state     bundle=control
#pragma HLS INTERFACE s_axilite port=expectation     bundle=control
    ComplexQ state[Config<3>::DIM];

#pragma HLS DATAFLOW
    
    qaoaStep_hls<3, 1>(state, d, gamma, beta);

    uint32_t dummy;
    *expectation = get_best_state
        ? expectation_cost<3>(state, d, best_state)
        : expectation_cost<3>(state, d, &dummy);
}


// template qfix costHamiltonian<3>(uint32_t s, const qfix d[3][3]);
// template int build_feasible_superposition<3>(ComplexQ state[Config<3>::DIM]);
// template void applyCost_hls<3>(ComplexQ state[Config<3>::DIM], const qfix d[3][3], qfix gamma);
// template void applyMixer_hls<3>(ComplexQ state[Config<3>::DIM], qfix beta);
// template void qaoaStep_hls<3,2>(ComplexQ state[Config<3>::DIM],
//                                 const qfix d[3][3],
//                                 const qfix gamma[2],
//                                 const qfix beta[2]);
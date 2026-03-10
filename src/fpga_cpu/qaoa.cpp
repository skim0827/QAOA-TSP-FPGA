#include "qaoa.hpp"
#include <cmath>

template<int N_CITY>
qfix costHamiltonian(uint32_t s, const qfix d[N_CITY][N_CITY]){

#pragma HLS INLINE

    qfix H=0;
    qfix P=30;

    for(int i=0;i<N_CITY;i++)
        for(int j=0;j<N_CITY;j++)
            for(int t=0;t<N_CITY;t++){

#pragma HLS PIPELINE II=1

                int next_t=(t+1)%N_CITY;

                int Z_i_t   = Z_eigenvalue_uint(s,t*N_CITY+i);
                int Z_j_t_1 = Z_eigenvalue_uint(s,next_t*N_CITY+j);

                H += d[i][j]*qfix(0.25)*(qfix(1)-Z_i_t)*(qfix(1)-Z_j_t_1);
            }


    for(int i=0;i<N_CITY;i++){

        qfix X_sum=0;

        for(int t=0;t<N_CITY;t++)
            X_sum += qfix(0.5)*(qfix(1)-Z_eigenvalue_uint(s,t*N_CITY+i));

        H += P*(X_sum-qfix(1))*(X_sum-qfix(1));
    }


    for(int t=0;t<N_CITY;t++){

        qfix X_sum=0;

        for(int i=0;i<N_CITY;i++)
            X_sum += qfix(0.5)*(qfix(1)-Z_eigenvalue_uint(s,t*N_CITY+i));

        H += P*(X_sum-qfix(1))*(X_sum-qfix(1));
    }

    return H;
}


template<int N_CITY>
bool is_valid_onehot(uint32_t s){

#pragma HLS INLINE

    for(int t=0;t<N_CITY;t++){

        int ones=0;

        for(int i=0;i<N_CITY;i++)
            ones += ((s>>(t*N_CITY+i))&1u);

        if(ones!=1) return false;
    }

    for(int i=0;i<N_CITY;i++){

        int ones=0;

        for(int t=0;t<N_CITY;t++)
            ones += ((s>>(t*N_CITY+i))&1u);

        if(ones!=1) return false;
    }

    return true;
}


template<int N_CITY>
int build_feasible_superposition(
        ComplexQ state[Config<N_CITY>::DIM]){

#pragma HLS INLINE off

    const int DIM=Config<N_CITY>::DIM;

    int count=0;

    for(int s=0;s<DIM;s++)
        if(is_valid_onehot<N_CITY>(s))
            count++;

    const qfix norm = qfix(1)/hls::sqrt((qfix)count);

    for(int s=0;s<DIM;s++)
        state[s] = is_valid_onehot<N_CITY>(s)
            ? ComplexQ(norm,0)
            : ComplexQ(0,0);

    return count;
}
template<int N_CITY>
void precompute_cost_table(
        qfix H_table[Config<N_CITY>::DIM],
        const qfix d[N_CITY][N_CITY])
{

#pragma HLS INLINE off

    for(uint32_t s=0; s<Config<N_CITY>::DIM; s++){

#pragma HLS PIPELINE II=1

        H_table[s] = costHamiltonian<N_CITY>(s,d);
    }
}

template<int N_CITY>
void applyCost_hls(
        ComplexQ state[Config<N_CITY>::DIM],
        const qfix H_table[Config<N_CITY>::DIM],
        qfix gamma)
{

#pragma HLS INLINE off

    for(uint32_t s=0;s<Config<N_CITY>::DIM;s++){

#pragma HLS PIPELINE II=1

        qfix Hs = H_table[s];

        qfix ang = gamma*Hs;

        state[s] *= ComplexQ(hls::cos(ang),-hls::sin(ang));
    }
}


template<int N_CITY>
void applyMixer_hls(
        ComplexQ state[Config<N_CITY>::DIM],
        qfix beta){

#pragma HLS INLINE off

    const qfix c = hls::cos(qfix(2)*beta);
    const qfix s = hls::sin(qfix(2)*beta);

    const uint32_t slice_mask=(1u<<N_CITY)-1u;

    static ComplexQ bufA[Config<N_CITY>::DIM];
    static ComplexQ bufB[Config<N_CITY>::DIM];

#pragma HLS BIND_STORAGE variable=bufA type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bufB type=ram_2p impl=bram

    for(uint32_t k=0;k<Config<N_CITY>::DIM;k++){
        #pragma HLS PIPELINE II=1
        bufA[k]=state[k];
    }

    ComplexQ* state_cur=bufA;
    ComplexQ* state_next=bufB;

    for(int t=0;t<N_CITY;t++){

        int base=t*N_CITY;

        for(int i=0;i<N_CITY;i++)
            for(int j=i+1;j<N_CITY;j++){

                for(uint32_t k=0;k<Config<N_CITY>::DIM;k++){
                    #pragma HLS PIPELINE II=1
                    state_next[k]=state_cur[k];
                }
                uint32_t bit_i=1u<<(t*N_CITY+i);
                uint32_t bit_j=1u<<(t*N_CITY+j);
                uint32_t mask=bit_i|bit_j;

                for(uint32_t s_idx=0;s_idx<Config<N_CITY>::DIM;s_idx++){

#pragma HLS PIPELINE II=1

                    uint32_t slice=(s_idx>>base)&slice_mask;

                    if(slice==(1u<<i)){

                        uint32_t s_flip=s_idx^mask;

                        ComplexQ a=state_cur[s_idx];
                        ComplexQ b=state_cur[s_flip];

                        state_next[s_idx]  = ComplexQ(c,0)*a - ComplexQ(0,s)*b;
                        state_next[s_flip] = ComplexQ(0,-s)*a + ComplexQ(c,0)*b;
                    }
                }

                ComplexQ* tmp=state_cur;
                state_cur=state_next;
                state_next=tmp;
            }
    }

    for(uint32_t k=0;k<Config<N_CITY>::DIM;k++)
        state[k]=state_cur[k];
}

template<int N_CITY>
qfix expectation_cost(
        ComplexQ state[Config<N_CITY>::DIM],
        const qfix H_table[Config<N_CITY>::DIM],
        uint32_t* best_state)
{

#pragma HLS INLINE off

    qfix result=0;
    qfix max_prob=-1;
    uint32_t argmax=0;

    for(int s=0;s<Config<N_CITY>::DIM;s++){

#pragma HLS PIPELINE II=1

        qfix prob=state[s].re*state[s].re
                 +state[s].im*state[s].im;

        qfix Hs = H_table[s];

        result += prob * Hs;

        if(prob > max_prob){
            max_prob = prob;
            argmax = s;
        }
    }

    *best_state = argmax;

    return result;
}

template<int N_CITY,int P>
void qaoaStep_hls(
        ComplexQ state[Config<N_CITY>::DIM],
        const qfix d[N_CITY][N_CITY],
        const qfix gamma[P],
        const qfix beta[P],
        qfix H_table[Config<N_CITY>::DIM])
{

    build_feasible_superposition<N_CITY>(state);

    precompute_cost_table<N_CITY>(H_table,d);

    for(int p=0;p<P;p++){

        applyCost_hls<N_CITY>(state,H_table,gamma[p]);

        applyMixer_hls<N_CITY>(state,beta[p]);
    }
}

extern "C"
void qaoa_kernel(
        const qfix d[3][3],
        const qfix gamma[1],
        const qfix beta[1],
        bool get_best_state,
        uint32_t* best_state,
        qfix* expectation)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=d bundle=control
#pragma HLS INTERFACE s_axilite port=gamma bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=get_best_state bundle=control
#pragma HLS INTERFACE s_axilite port=best_state bundle=control
#pragma HLS INTERFACE s_axilite port=expectation bundle=control

#pragma HLS ARRAY_PARTITION variable=d complete dim=0

    ComplexQ state[Config<3>::DIM];
    qfix H_table[Config<3>::DIM];

#pragma HLS BIND_STORAGE variable=state type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=H_table type=ram_2p impl=bram
    qaoaStep_hls<3,1>(state,d,gamma,beta,H_table);

    uint32_t dummy;

    *expectation = get_best_state
        ? expectation_cost<3>(state,H_table,best_state)
        : expectation_cost<3>(state,H_table,&dummy);
}
#include "xparameters.h"
#include "xuartps.h"
#include <stdio.h>
#include <stdint.h>
#include "xiltimer.h"
#include "xuartps_hw.h"
//#include <nlopt.hpp>
#define QAOA_BASE XPAR_QAOA_KERNEL_0_BASEADDR
volatile uint32_t *qaoa = (uint32_t*)QAOA_BASE;

// ----- float -> ap_fixed<32,12> -----
static inline uint32_t float_to_fixed(float x){
    return (uint32_t)(x * (1 << 20));   // fractional bits = 20
}

// ----- ap_fixed<32,12> -> float -----
static inline float fixed_to_float(uint32_t x){
    return ((float)x) / (1 << 20);
}
// ---------------------------
// UART 
// ---------------------------

XUartPs uart;
void uart_init(){ 
    XUartPs_Config *cfg;
    cfg = XUartPs_LookupConfig(0);
    XUartPs_CfgInitialize(&uart, cfg, cfg->BaseAddress);
    XUartPs_SetBaudRate(&uart, 115200);
}

void uart_recv_PC(char *buf){ 
    int i = 0;
    // PC -> ARM wait until newline
    while (1) {
        if (XUartPs_IsReceiveData(uart.Config.BaseAddress)) {
            buf[i++] = XUartPs_ReadReg(uart.Config.BaseAddress,
                                    XUARTPS_FIFO_OFFSET);
            if (buf[i-1] == '\n') break;
        }
    }
    buf[i] = 0 ; 
}

void run_fpga(){
    qaoa[0] = 1; // ap_start 
    while((qaoa[0] & 0x2) == 0); // wait until ap_done
}

void uart_send_to_pc(float v){
    char buf[32]; 
    sprintf(buf, "%f\n", v); 
    for (int i = 0; buf[i]; i++){ 
        XUartPs_WriteReg(uart.Config.BaseAddress, XUARTPS_FIFO_OFFSET, buf[i]); 
    }    
}

int main() {

    uart_init();

    char buf[64];
    float gamma, beta;

    while(1){

    // 1. receive from PC
        uart_recv_PC(buf); 
        sscanf(buf, "%f %f", &gamma, &beta);

    // 2. write to fpga
        qaoa[6] = float_to_fixed(gamma);
        qaoa[8] = float_to_fixed(beta);    
        
    // 3. run fpga 
        run_fpga();

    // 4. read result
        float expectation = fixed_to_float(qaoa[4]);

    // 5. send back to PC 
        uart_send_to_pc(expectation);
    }
    
    // ---------------------------
    // time measurement 
    // ---------------------------
    // XTime t0, t1;
    // XTime_GetTime(&t0);
    // reg[0] = 1;
    // while ((reg[0] & 0x2) == 0);   // wait ap_done
    // XTime_GetTime(&t1);
    // double fpga_ms = (double)(t1 - t0) / (COUNTS_PER_SECOND / 1000.0);
    // printf("FPGA time = %f ms\n", fpga_ms);


    return 0;
}

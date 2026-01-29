#include "qaoa.hpp"
# include <iostream>
#include <cmath>
#include <limits>
#include <bitset>
#include <nlopt.hpp>
#include <vector>
// screen /dev/ttyUSB0 115200
// picocom /dev/ttyUSB0 -b 115200
//ls /dev/ttyUSB*

// =========================================================
// uart helper 
// =========================================================
#include <fcntl.h> 
#include <unistd.h>
#include <termios.h>

int uart_fd;
void uart_init(const char* dev = "/dev/ttyUSB1")
{
    uart_fd = open(dev, O_RDWR | O_NOCTTY | O_SYNC);
    if (uart_fd < 0) {
        perror("uart open failed");
        exit(1);
    }

    struct termios tty{};
    if (tcgetattr(uart_fd, &tty) != 0) {
        perror("tcgetattr");
        exit(1);
    }

    // Baud rate: MUST match Zybo
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    // 8N1
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);

    // Raw mode (VERY important)
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    if (tcsetattr(uart_fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        exit(1);
    }

    // Clear any garbage in UART buffers
    tcflush(uart_fd, TCIOFLUSH);
}

// =========================================================
// calling FPGA
// =========================================================
double fpga_qaoa_eval(double gamma, double beta){
    char buf[64]; 
    // send gamma and beta 
    sprintf(buf, "%f %f\n", gamma, beta);
    write(uart_fd, buf, strlen(buf));
    // read expectation 
    int n = read(uart_fd, buf, sizeof(buf));

    if (n <= 0 ){ 
        printf("uart read fialed\n");
        exit(1);
    }
    buf[n] = 0; // -- string terminator '\n'

    return atof(buf); // -- text to double
}

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
// NLopt objective
// =========================================================
double objective_function(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    (void)grad; (void)data;
    
    // expectation = qaoa_kernel(d, gamma, beta, false, &dummy);
    double expectation = fpga_qaoa_eval(x[0], x[1]);

    return expectation;
}

// ------------------- Debug wrapper -------------------
void qaoa_kernel_debug(const double d[3][3],
                       const double gamma[1],
                       const double beta[1],
                       double* expectation_out,
                       uint32_t* best_state_out)
{
    uint32_t best_state = 0; 

    double expectation = qaoa_kernel(d, gamma, beta, true, &best_state);

    *expectation_out = expectation ; 
    *best_state_out = best_state ;

    double min_energy_br;
    uint32_t best_state_br = brute_force<3>(d, min_energy_br);

    std::cout << "Best bitstring BRUTE (9 bits): "
              << std::bitset<9>(best_state_br) << std::endl;
    std::cout << "Best bitstring (QAOA p=2)  : "
              << std::bitset<9>(best_state) << std::endl;}

// =========================================================
// main 
// =========================================================
int main() {

    uart_init();
    std::cout << "============== uart initialised ==================\n";

    // --- Set up NLopt (Nelder–Mead) ---
    nlopt::opt opt(nlopt::LN_NELDERMEAD, 2); // parameters to optimise
    opt.set_lower_bounds({0.0, 0.0});
    opt.set_upper_bounds({M_PI, M_PI});
    opt.set_min_objective(objective_function, nullptr);
    opt.set_xtol_rel(1e-4);
    
    // --- Run optimizer ---
    std::cout << "============== Running optimiser==================\n";
    static int iteration = 0; 

    std::vector<double> x = {0.5, 0.5};
    double min_expectation ;
    nlopt::result result = opt.optimize(x, min_expectation);

    std::cout << "============== Optimisation finished==================\n";
    std::cout << "gamma = " << x[0] << "\n";
    std::cout << "beta  = " << x[1] << "\n";
    std::cout << "expectation = " << min_expectation << "\n";    

    close(uart_fd);
    return 0;
}


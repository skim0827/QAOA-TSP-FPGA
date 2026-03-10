tb : qaoa.cpp tb.cpp qaoa.hpp 
	g++ -std=c++17 -Wall qaoa.cpp tb.cpp -o tb 

op : qaoa.cpp optimisation.cpp qaoa.hpp
	g++ -std=c++17 -Wall qaoa.cpp optimisation.cpp -o optimisation -lnlopt
clean: 
	rm -f tb optimisation

run_fpga : tb
	./tb 

run_cpu : op
	./optimisation
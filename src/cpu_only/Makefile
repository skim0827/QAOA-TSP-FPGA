op : qaoa_ref.cpp optimisation_ref.cpp qaoa_ref.hpp
	g++ -std=c++17 -Wall qaoa_ref.cpp optimisation_ref.cpp -o optimisation_ref -lnlopt
clean: 
	rm -f optimisation_ref

run_cpu : op
	./optimisation_ref
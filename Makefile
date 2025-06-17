FFTW=/home/99r/collab/paradyse-master/venv

libsfac.so: sfac.cpp
	g++ -fopenmp -std=c++11 -fPIC -shared -I$(FFTW)/include -Wl,-rpath,$(FFTW)/lib -L$(FFTW)/lib -o $@ $^ -lfftw3

test: test.cpp sfac.cpp
	g++ -I$(FFTW)/include -Wl,-rpath,$(FFTW)/lib -L$(FFTW)/lib -o $@ $^ -lfftw3

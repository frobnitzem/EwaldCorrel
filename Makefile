FFTW=/usr/local/spack/opt/spack/darwin-bigsur-skylake/apple-clang-13.0.0/fftw-3.3.10-2lf6fmlp7gnxvj67sbmnhbqbwvuqpxyq

libsfac.so: sfac.cpp
	g++ -fPIC -shared -I$(FFTW)/include -Wl,-rpath,$(FFTW)/lib -L$(FFTW)/lib -o $@ $^ -lfftw3

test: test.cpp sfac.cpp
	g++ -I$(FFTW)/include -Wl,-rpath,$(FFTW)/lib -L$(FFTW)/lib -o $@ $^ -lfftw3

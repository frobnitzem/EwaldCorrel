# Ewald reciprocal space calculation

This library provides Fourier-space calculations of the
approximate structure factor using the smooth particle
mesh Ewald method.  For details, see the included tex file.


## Building

To build, fix your fftw3 location in the Makefile and run make.

Example build and install sequence:

```bash
FFTW=/usr/local make
```


## Testing

Run either the python or C++ tests.

```shell
python3 test.py
make test
```

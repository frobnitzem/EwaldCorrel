libsfac.so: sfac.c
	gcc-fsf-5 -I/sw/include -L/sw/lib -shared -o libsfac.so sfac.c -lfftw3

libsfac.so: sfac.c
	gcc -fPIC -shared -o libsfac.so sfac.c -lfftw3

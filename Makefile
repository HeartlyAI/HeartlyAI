build:
	g++ -shared -O3 -fPIC \
		-o vendor/hamilton.so vendor/hamilton/python.c \
		vendor/hamilton/qrsdet.cpp vendor/hamilton/qrsfilt.cpp \
		-I /usr/include/python* -I /usr/lib/python3*/site-packages/numpy/core/include

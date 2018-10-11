BIN_DIR=../bin

CC = g++
CFLAGS = -std=c++11 -pthread -Wall -Ofast -mtune=native -march=native -mfpmath=sse -funroll-loops
DCFLAGS = -std=c++11 -pthread -Wall -O0

all: main.cpp options.cpp utils.cpp vocab.cpp model.cpp vector.cpp matrix.cpp
	$(CC) main.cpp options.cpp utils.cpp vocab.cpp model.cpp vector.cpp matrix.cpp -o ${BIN_DIR}/sv4d $(CFLAGS)

debug: main.cpp options.cpp utils.cpp vocab.cpp model.cpp vector.cpp matrix.cpp
	$(CC) main.cpp options.cpp utils.cpp vocab.cpp model.cpp vector.cpp matrix.cpp -o ${BIN_DIR}/sv4d $(DCFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf sv4d; popd

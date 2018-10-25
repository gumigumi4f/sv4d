BINDIR=../bin

CXX = c++
CXXFLAGS = -std=c++11 -pthread -Wall -Wextra -Wno-unused-result
OBJS = $(BINDIR)/utils.o $(BINDIR)/vector.o $(BINDIR)/matrix.o $(BINDIR)/options.o $(BINDIR)/vocab.o $(BINDIR)/model.o

all: CXXFLAGS += -Ofast -march=native -mtune=native -funroll-loops -flto
all: sv4d

debug: CXXFLAGS += -O0 -g -fno-inline
debug: sv4d

$(BINDIR)/utils.o: utils.cpp utils.hpp
	$(CXX) $(CXXFLAGS) -c utils.cpp -o $(BINDIR)/utils.o

$(BINDIR)/vector.o: vector.cpp vector.hpp utils.hpp
	$(CXX) $(CXXFLAGS) -c vector.cpp -o $(BINDIR)/vector.o

$(BINDIR)/matrix.o: matrix.cpp matrix.hpp vector.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp -o $(BINDIR)/matrix.o

$(BINDIR)/options.o: options.cpp options.hpp
	$(CXX) $(CXXFLAGS) -c options.cpp -o $(BINDIR)/options.o

$(BINDIR)/vocab.o: vocab.cpp vocab.hpp options.hpp utils.hpp
	$(CXX) $(CXXFLAGS) -c vocab.cpp -o $(BINDIR)/vocab.o

$(BINDIR)/model.o: model.cpp model.hpp utils.hpp vector.hpp matrix.hpp options.hpp vocab.hpp
	$(CXX) $(CXXFLAGS) -c model.cpp -o $(BINDIR)/model.o

sv4d: $(OBJS) main.cpp
	$(CXX) $(CXXFLAGS) $(OBJS) main.cpp -o $(BINDIR)/sv4d

clean:
	pushd $(BINDIR) && rm -rf *.o sv4d; popd

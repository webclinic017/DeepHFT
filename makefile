COMPILER=g++
VERSION=-std=c++11

output: main.o trend_hft.o data.o linear.o dnn.o
	$(COMPILER) $(VERSION) main.o trend_hft.o data.o linear.o dnn.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

# --- #

trend_hft.o: ./src/trend_hft.cpp
	$(COMPILER) $(VERSION) -c ./src/trend_hft.cpp

# --- #

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

linear.o: ./src/linear.cpp
	$(COMPILER) $(VERSION) -c ./src/linear.cpp

dnn.o: ./src/dnn.cpp
	$(COMPILER) $(VERSION) -c ./src/dnn.cpp

COMPILER=g++
VERSION=-std=c++11

output: main.o trend.o data.o linear.o dnn.o
	$(COMPILER) $(VERSION) main.o trend.o data.o linear.o dnn.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

# --- #

trend.o: ./src/trend.cpp
	$(COMPILER) $(VERSION) -c ./src/trend.cpp

# --- #

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

linear.o: ./src/linear.cpp
	$(COMPILER) $(VERSION) -c ./src/linear.cpp

dnn.o: ./src/dnn.cpp
	$(COMPILER) $(VERSION) -c ./src/dnn.cpp

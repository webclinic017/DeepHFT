COMPILER=g++
VERSION=-std=c++20

output: main.o trend.o data.o linear.o dnn.o checkpoint.o
	$(COMPILER) $(VERSION) main.o trend.o data.o linear.o dnn.o checkpoint.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

trend.o: ./src/trend.cpp
	$(COMPILER) $(VERSION) -c ./src/trend.cpp

# --- #

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

linear.o: ./src/linear.cpp
	$(COMPILER) $(VERSION) -c ./src/linear.cpp

dnn.o: ./src/dnn.cpp
	$(COMPILER) $(VERSION) -c ./src/dnn.cpp

checkpoint.o: ./src/checkpoint.cpp
	$(COMPILER) $(VERSION) -c ./src/checkpoint.cpp

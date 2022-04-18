COMPILER=g++
VERSION=-std=c++11

output: main.o train.o run.o data.o linear.o dnn.o
	$(COMPILER) $(VERSION) main.o train.o run.o data.o linear.o dnn.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

# --- #

train.o: ./src/train.cpp
	$(COMPILER) $(VERSION) -c ./src/train.cpp

run.o: ./src/run.cpp
	$(COMPILER) $(VERSION) -c ./src/run.cpp

# --- #

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

linear.o: ./src/linear.cpp
	$(COMPILER) $(VERSION) -c ./src/linear.cpp

dnn.o: ./src/dnn.cpp
	$(COMPILER) $(VERSION) -c ./src/dnn.cpp

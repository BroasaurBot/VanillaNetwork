CC = g++
TARGET = vanillaNet
CFLAGS = -Wall -Werror -std=c++11

all: $(TARGET)

$(TARGET): main.o dataManager.o Helper.o Layers.o vanillaNetwork.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o dataManager.o Layers.o Helper.o vanillaNetwork.o

runner: runner.cpp dataManager.o Helper.o Layers.o vanillaNetwork.o
	$(CC) $(CFLAGS) -c runner.cpp -o main.o
	$(CC) $(CFLAGS) -o vanillaRunner main.o dataManager.o Layers.o Helper.o vanillaNetwork.o

test: clean all
	./vanillaNet

main.o:
	$(CC) $(CFLAGS) -c main.cpp

dataManager.o:
	$(CC) $(CFLAGS) -c dataManager.cpp

Layers.o:
	$(CC) $(CFLAGS) -c Layers.cpp

Helper.o:
	$(CC) $(CFLAGS) -c Helper.cpp

vanillaNetwork.o:
	$(CC) $(CFLAGS) -c vanillaNetwork.cpp

clean:
	rm *.o vanillaNet
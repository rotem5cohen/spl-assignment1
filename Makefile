
CXX = g++
CXXFLAGS = -Wall -Weffc++ -std=c++11 -Iinclude

SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(patsubst src/%.cpp,bin/%.o,$(SOURCES))
TARGET = bin/server

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f bin/*.o bin/server

.PHONY: all clean

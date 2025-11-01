# Simple Makefile for header-only matrixlib

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Iinclude

EXAMPLE := examples/main

all: $(EXAMPLE)

$(EXAMPLE): examples/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(EXAMPLE)

.PHONY: all clean

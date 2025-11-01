CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude

SRC_DIR := src
TEST_DIR := tests
EXAMPLES_DIR := examples

TEST_SRC := $(TEST_DIR)/catch_amalgamated.cpp $(TEST_DIR)/test_matrix.cpp
TEST_BIN := $(TEST_DIR)/test_matrix

EXAMPLE_BIN := $(EXAMPLES_DIR)/example

all: test example

test: $(TEST_BIN)
	@echo "Running tests..."
	@$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

example: $(EXAMPLES_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $(EXAMPLE_BIN)

clean:
	rm -f $(TEST_BIN) $(EXAMPLE_BIN)

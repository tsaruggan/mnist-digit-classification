# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11

# Directories
SRC_DIR = src
EXE_DIR = .
TARGET = $(EXE_DIR)/nn

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)

# Default target compiles and runs
all: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(EXE_DIR)
	$(CXX) $(OBJS) -o $(TARGET)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

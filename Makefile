# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 \
            -I/usr/local/Cellar/opencv/4.10.0_12/include/opencv4
LDFLAGS = -L/usr/local/Cellar/opencv/4.10.0_12/lib \
          -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

# Directories
SRC_DIR = src
EXE_DIR = .
TARGET = $(EXE_DIR)/classifier  # Executable name

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)

# Default target: build everything
all: $(TARGET)

# Train target: build and run training
train: $(TARGET)
	./$(TARGET) train

# Run target: build and run prediction service
run: $(TARGET)
	./$(TARGET) run

# Compile and link the object files into the executable
$(TARGET): $(OBJS)
	mkdir -p $(EXE_DIR)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile the source files into object files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files and the executable
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

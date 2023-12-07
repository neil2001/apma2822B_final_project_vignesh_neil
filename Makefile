# Makefile for CUDA project

# Compiler
CC := nvcc

# Flags
CFLAGS := -std=c++11
CUDAFLAGS := -arch=sm_60

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Source files
SRC := $(wildcard $(SRC_DIR)/*.cu)
OBJ := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC))

# Executable name
TARGET := $(BIN_DIR)/cuda_project

# Build rule
all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)



# Makefile for C++ project

# CXX = g++
# CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic
# LDFLAGS = 

# SRCDIR = src
# OBJDIR = obj
# BINDIR = bin

# # List of source files
# SOURCES = $(wildcard $(SRCDIR)/**/*.cpp) \
#           $(SRCDIR)/main_cpu.cpp

# # Generate list of object files from source files
# OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

# # Name of the executable
# EXECUTABLE = my_program

# # Path to the submodule
# TQDM_SUBMODULE_PATH = submodules/tqdm.cpp/include

# # Include directory for the submodule
# INCLUDE_DIRS = -I $(TQDM_SUBMODULE_PATH)

# all: $(EXECUTABLE)

# # Rule to link object files into the executable
# $(EXECUTABLE): $(OBJECTS)
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDE_DIRS) -o $@ $^

# # Rule to compile source files into object files
# $(OBJDIR)/%.o: $(SRCDIR)/%.cpp
# 	@mkdir -p $(@D)
# 	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

# clean:
# 	rm -rf $(OBJDIR) $(BINDIR)

# .PHONY: all clean



# CUDA_PATH     ?= /usr/local/cuda
# HOST_COMPILER  = g++
# NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# # select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
# #NVCC_DBG       =

# NVCCFLAGS      = $(NVCC_DBG) -m64
# GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

# cudart: cudart.o
# 	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

# cudart.o: main.cu
# 	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

# out.ppm: cudart
# 	rm -f out.ppm
# 	./cudart > out.ppm

# out.jpg: out.ppm
# 	rm -f out.jpg
# 	ppmtojpeg out.ppm > out.jpg

# profile_basic: cudart
# 	nvprof ./cudart > out.ppm

# # use nvprof --query-metrics
# profile_metrics: cudart
# 	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

# clean:
# 	rm -f cudart cudart.o out.ppm out.jpg
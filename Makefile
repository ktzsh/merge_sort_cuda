NVCC = nvcc
GCC = gcc
FLAGS = -Xcompiler -fPIC
TARGET  = cuMergesort.so
CHECKER = checker

all: $(TARGET) $(CHECKER)
$(TARGET): cuMergesort.cu
	$(NVCC) $(FLAGS) -o $(TARGET) -shared cuMergesort.cu
$(CHECKER): checker.c
	$(GCC) -fPIC -o $(CHECKER) checker.c cuMergesort.so

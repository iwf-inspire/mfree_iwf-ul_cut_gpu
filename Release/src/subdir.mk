################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/benchmarks.cpp \
../src/constants_structs.cpp \
../src/debug.cpp \
../src/geometric_primitives.cpp \
../src/mfree_gpu.cpp \
../src/tool.cpp \
../src/types.cpp \
../src/vtk_writer.cpp 

CU_SRCS += \
../src/actions_gpu.cu \
../src/grid.cu \
../src/grid_gpu_green.cu \
../src/grid_gpu_rothlin.cu \
../src/interactions_gpu.cu \
../src/leap_frog.cu \
../src/output.cu \
../src/particle_gpu.cu \
../src/tool_gpu.cu \
../src/tool_wear.cu 

CU_DEPS += \
./src/actions_gpu.d \
./src/grid.d \
./src/grid_gpu_green.d \
./src/grid_gpu_rothlin.d \
./src/interactions_gpu.d \
./src/leap_frog.d \
./src/output.d \
./src/particle_gpu.d \
./src/tool_gpu.d \
./src/tool_wear.d 

OBJS += \
./src/actions_gpu.o \
./src/benchmarks.o \
./src/constants_structs.o \
./src/debug.o \
./src/geometric_primitives.o \
./src/grid.o \
./src/grid_gpu_green.o \
./src/grid_gpu_rothlin.o \
./src/interactions_gpu.o \
./src/leap_frog.o \
./src/mfree_gpu.o \
./src/output.o \
./src/particle_gpu.o \
./src/tool.o \
./src/tool_gpu.o \
./src/tool_wear.o \
./src/types.o \
./src/vtk_writer.o 

CPP_DEPS += \
./src/benchmarks.d \
./src/constants_structs.d \
./src/debug.d \
./src/geometric_primitives.d \
./src/mfree_gpu.d \
./src/tool.d \
./src/types.d \
./src/vtk_writer.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



# 1. 
略

# 2.
cudaDeviceReset()显示释放和清空当前进程中所有资源，移除该函数后，编译运行无法正常输出有关Device的代码

# 3.
cudaDeviceSynchronize()替换cudaDeviceReset(),可以正常输出代码

# 4.
编译命令行移除设备架构标志无影响

# 5.
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
.cu
.c
.cc .cxx .cpp
.ptx
.cubin
.fabin
.o .obj
.a .lib
.res
.so

# 6.
exp6.cpp
当前线程在block的索引：threadIdx.x threadIdx.y threadIdx.z
当前block 在grid的索引：blockIdx.x blockIdx.y blockIdx.z
每个block的线程数：blockDim.x

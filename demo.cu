#include "shared_array.hpp"
#include <thrust/device_vector.h>
#include <cstdio>
#include <iostream>

__global__ void single_reader_hazard(thrust::device_ptr<int> result)
{
  shared_array<128> smem;

  smem[threadIdx.x] = 1;

  // XXX missing barrier here

  if(threadIdx.x == 0)
  {
    // read after write
    int val = smem[blockDim.x - threadIdx.x - 1];

    *result = val;
  }
}

__global__ void multiple_writers_hazard(thrust::device_ptr<int> result)
{
  shared_array<128> smem;

  // XXX all threads write to the same location
  smem[0] = threadIdx.x;

  smem.barrier();

  if(threadIdx.x == 0)
  {
    *result = smem[0];
  }
}

__global__ void race_free_reduction(thrust::device_ptr<int> result)
{
  shared_array<128> smem;

  smem[threadIdx.x] = 1;

  smem.barrier();

  unsigned int n = 128;
  while(n > 1)
  {
    unsigned int half = n / 2;

    if(threadIdx.x < half)
    {
      smem[threadIdx.x] = smem[threadIdx.x] + smem[n - threadIdx.x - 1];
    }

    smem.barrier();

    n = n - half;
  }

  if(threadIdx.x == 0)
  {
    *result = smem[0];
  }
}

int main()
{
  thrust::device_vector<int> vec(1);

  race_free_reduction<<<1,128>>>(vec.data());

  std::cout << "race_free_reduction result is " << vec[0] << std::endl;

  single_reader_hazard<<<1,128>>>(vec.data());

  std::cout << "single_reader_hazard result is " << vec[0] << std::endl;

  multiple_writers_hazard<<<1,128>>>(vec.data());

  std::cout << "multiple_writers_hazard result is " << vec[0] << std::endl;

  return 0;
}


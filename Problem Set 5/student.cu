/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible
  int myId = blockDim.x * blockIdx.x + threadIdx.x;
  if(myId >= numVals) return;

  unsigned int bin = vals[myId];
  atomicAdd(&histo[bin],1);
  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  dim3 blockSize(256, 1, 1);
  dim3 gridSize((numElems + blockSize.x -1)/ blockSize.x, 1, 1);
  unsigned int * d_histogram;
  unsigned int size = sizeof(unsigned int) * numBins;
  checkCudaErrors(cudaMalloc(&d_histogram, size));
  checkCudaErrors(cudaMemcpy(d_histogram, d_histo, size, cudaMemcpyDeviceToDevice));
  yourHisto<<<gridSize, blockSize>>>(d_vals, d_histogram, numElems);
  checkCudaErrors(cudaMemcpy(d_histo, d_histogram, size, cudaMemcpyDeviceToDevice));

  //if you want to use/launch more than one kernel,
  //feel free

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

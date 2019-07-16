//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void copy_data(unsigned int * d_histogram, unsigned int * d_scan){
        d_scan[1] = d_histogram[0];
        // printf("d_scan[1] : %d \n", d_scan[1]);
}

__global__ void histogram(
        unsigned int * d_histogram, 
        unsigned int *d_vals_src, 
        unsigned int mask, 
        unsigned int numElems, 
        unsigned int i)
{
    int myId = blockDim.x * blockIdx.x + threadIdx.x;
//     int tid = threadIdx.x;
    
    if(myId >= numElems){
        return;
    }
    unsigned int bin = (d_vals_src[myId] & mask) >> i;
    atomicAdd(&d_histogram[bin],1);
}

__global__ void scan_across_block(unsigned int * d_sums, unsigned int * d_sums_cdf, unsigned int numElems){
    extern __shared__ unsigned int sdata[];

    int myId = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[ 2 * tid ] = 0;
    sdata[ 2 * tid + 1 ] = 0;
    if(2 * myId < numElems){
        // unsigned int bin = (d_vals_src[2 * myId]  & mask) >> i;
        sdata[2 * tid ] = d_sums[2 * myId];
    } 

    if (2 * myId +1 < numElems){
        // unsigned int bin = (d_vals_src[2 * myId +1 ]  & mask) >> i;
        sdata[2 * tid + 1] = d_sums[2 *myId +1];
    }
    __syncthreads();

    //reduce
    for(unsigned int s = 1 ; s <= blockDim.x; s<<=1){
        if(tid < blockDim.x /s){
            sdata[s *(2*tid + 2) -1] += sdata[s*(2*tid +1) -1];
        }
        __syncthreads();
    }

    if(tid == 0){
        sdata[2 * blockDim.x -1] =  0;
    }

    //down sweep
    for( unsigned int s = blockDim.x ; s > 0 ; s >>=1){
        if(tid < blockDim.x /s){
            unsigned int tmp = sdata[s *(2*tid + 2) -1];
            sdata[s*(2*tid +2) -1] += sdata[s*(2*tid +1) -1];
            sdata[s*(2*tid +1) -1] = tmp; 
        }
        __syncthreads();
    }

    if(2 * myId < numElems){
        d_sums_cdf[2 * myId] = sdata[2 * tid];
    } 

    if (2 * myId +1 < numElems){
        d_sums_cdf[2 * myId + 1] = sdata[2 * tid + 1];
    }
}

__global__ void compute_outputPos(
        unsigned int * d_vals_src, 
        unsigned int *d_pos_dst, 
        unsigned int * d_cdf,
        unsigned int * d_scan,
        unsigned int * d_sums_cdf,
        unsigned int numElems,
        unsigned int mask, 
        unsigned int i)
{
    int myId = blockDim.x * blockIdx.x + threadIdx.x;
    if(myId >=numElems ) return;
    unsigned int bin = (d_vals_src[myId] &mask)>> i;
    unsigned int sum = d_sums_cdf[blockIdx.x] ;
    // for(int j=0; j< blockIdx.x; j++){
    //     sum += d_sums[j];
    // }
    // printf("sum: %d\n",sum);
    if(bin == 0){
        d_pos_dst[myId] = d_scan[0] +  myId - (sum+d_cdf[myId]);
    }else{
        d_pos_dst[myId] = d_scan[1] + sum + d_cdf[myId];
    }   
    // printf("d_pos_dst[myid] : %d \n", d_pos_dst[myId]);
}



__global__ void scan_per_block(unsigned int * d_vals_src, unsigned int * d_cdf, unsigned int * d_sums,unsigned int numElems,unsigned int mask, unsigned int i)
{
    extern __shared__ unsigned int sdata[];

    int myId = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[ 2 * tid ] = 0;
    sdata[ 2 * tid + 1 ] = 0;
    if(2 * myId < numElems){
        unsigned int bin = (d_vals_src[2 * myId]  & mask) >> i;
        sdata[2 * tid ] = bin;
    } 

    if (2 * myId +1 < numElems){
        unsigned int bin = (d_vals_src[2 * myId +1 ]  & mask) >> i;
        sdata[2 * tid + 1] = bin;
    }
    __syncthreads();

    //reduce
    for(unsigned int s = 1 ; s <= blockDim.x; s<<=1){
        if(tid < blockDim.x /s){
            sdata[s *(2*tid + 2) -1] += sdata[s*(2*tid +1) -1];
        }
        __syncthreads();
    }

    if(tid == 0){
        d_sums[blockIdx.x] =  sdata[2 * blockDim.x -1];
        sdata[2 * blockDim.x -1] =  0;
    }

    //down sweep
    for( unsigned int s = blockDim.x ; s > 0 ; s >>=1){
        if(tid < blockDim.x /s){
            unsigned int tmp = sdata[s *(2*tid + 2) -1];
            sdata[s*(2*tid +2) -1] += sdata[s*(2*tid +1) -1];
            sdata[s*(2*tid +1) -1] = tmp; 
        }
        __syncthreads();
    }

    if(2 * myId < numElems){
        d_cdf[2 * myId] = sdata[2 * tid];
    } 

    if (2 * myId +1 < numElems){
        d_cdf[2 * myId + 1] = sdata[2 * tid + 1];
    }
    
}

__global__ void gather(
        unsigned int * d_vals_src, 
        unsigned int * d_vals_dst, 
        unsigned int * d_pos_src, 
        unsigned int * d_pos_dst, 
        unsigned int * d_out_pos,
        unsigned int numElems)
{
    int myId = blockDim.x * blockIdx.x + threadIdx.x;

    if(myId >= numElems){
        return;
    }

//     unsigned int bin = (d_vals_src[myId]& mask) >> i;
    d_vals_dst[d_out_pos[myId]] = d_vals_src[myId];
    d_pos_dst[d_out_pos[myId]] = d_pos_src[myId];
//     atomicAdd(&(d_scan[bin]), 1);
}
    
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO 
  //PUT YOUR SORT HERE
  dim3 blockSize(256, 1 , 1);
  dim3 gridSize((numElems + blockSize.x -1)/ blockSize.x, 1, 1 );
//   printf(" numElems: %d gridSize.x : %d\n", numElems,gridSize.x);
  dim3 blockS(blockSize.x /2, 1, 1);
  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int *d_vals_src ;
  unsigned int *d_pos_src ;
  unsigned int *d_vals_dst ;
  unsigned int *d_pos_dst ;
  unsigned int * d_out_pos;
  size_t size = sizeof(unsigned int ) * numElems;
  checkCudaErrors(cudaMalloc(&d_vals_src, size));
  checkCudaErrors(cudaMalloc(&d_pos_src, size));
  checkCudaErrors(cudaMalloc(&d_vals_dst, size));
  checkCudaErrors(cudaMalloc(&d_pos_dst, size));
  checkCudaErrors(cudaMalloc(&d_out_pos, size));

  checkCudaErrors(cudaMemcpy(d_vals_src , d_inputVals, size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_pos_src , d_inputPos, size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_vals_dst , d_outputVals, size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_pos_dst , d_outputPos, size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_out_pos , d_outputPos, size, cudaMemcpyDeviceToDevice));
  unsigned int * d_histogram;
  unsigned int * d_scan;
  unsigned int * d_cdf;
  unsigned int * d_sums;
  unsigned int * d_sums_cdf;
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int)*numElems));
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int)*(gridSize.x)));
  checkCudaErrors(cudaMalloc(&d_sums_cdf, sizeof(unsigned int)*(gridSize.x)));

  unsigned int nextPow = gridSize.x ;
  nextPow--;
  nextPow = (nextPow >> 1) | nextPow;
  nextPow = (nextPow >> 2) | nextPow;
  nextPow = (nextPow >> 4) | nextPow;
  nextPow = (nextPow >> 8) | nextPow;
  nextPow = (nextPow >> 16) | nextPow;
  nextPow++;
  dim3 blockSize1(nextPow/2, 1, 1);

//   checkCudaErrors(cudaMemset(d_histogram, 0 , sizeof(unsigned int)*numBins));

  for(unsigned int i = 0 ; i< 32; i+=numBits){
      unsigned int mask = (numBins -1) << i;
      checkCudaErrors(cudaMemset(d_histogram, 0 , sizeof(unsigned int)*numBins));
      checkCudaErrors(cudaMemset(d_scan, 0 , sizeof(unsigned int)*numBins));
      checkCudaErrors(cudaMemset(d_cdf, 0 , sizeof(unsigned int)*numElems));
      checkCudaErrors(cudaMemset(d_sums, 0 , sizeof(unsigned int)*(gridSize.x)));
      checkCudaErrors(cudaMemset(d_sums_cdf, 0 , sizeof(unsigned int)*(gridSize.x)));

//       memset(h_scan, 0, sizeof(unsigned int ) *numBins);
//       memset(h_histogram, 0, sizeof(unsigned int ) *numBins);
      
      scan_per_block<<<gridSize, blockS, sizeof(unsigned int)* blockSize.x>>>(d_vals_src, d_cdf, d_sums, numElems,mask ,i);
      scan_across_block<<<1, blockSize1, sizeof(unsigned int ) * nextPow>>>(d_sums, d_sums_cdf, gridSize.x);

      histogram<<<gridSize, blockSize>>>(d_histogram, d_vals_src, mask, numElems, i);
    //   checkCudaErrors(cudaMemcpy(d_scan,d_histogram +1 , sizeof(unsigned int), cudaMemcpyDeviceToDevice));
      copy_data<<<1,1>>>(d_histogram,d_scan);
//       checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, numBins* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    //   compute_outputPos<<<gridSize,blockSize>>>(d_vals_src, d_out_pos,d_cdf, d_scan,d_sums,numElems, mask ,i);
      compute_outputPos<<<gridSize,blockSize>>>(d_vals_src, d_out_pos,d_cdf, d_scan,d_sums_cdf,numElems, mask ,i);
      
      gather<<<gridSize, blockSize>>>(d_vals_src,d_vals_dst,d_pos_src,d_pos_dst,d_out_pos ,numElems);
      
      std::swap(d_vals_dst, d_vals_src);
      std::swap(d_pos_dst, d_pos_src);
      
  }

  checkCudaErrors(cudaMemcpy(d_vals_dst, d_vals_src,size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_pos_dst, d_pos_src, size , cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemcpy( d_inputVals,d_vals_src , size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_inputPos,d_pos_src , size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_outputVals,d_vals_dst , size, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_outputPos,d_pos_dst ,size, cudaMemcpyDeviceToDevice));


  checkCudaErrors(cudaFree(d_vals_src));
  checkCudaErrors(cudaFree(d_vals_dst));
  checkCudaErrors(cudaFree(d_pos_src));
  checkCudaErrors(cudaFree(d_pos_dst));
  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_sums));
  checkCudaErrors(cudaFree(d_sums_cdf));
  checkCudaErrors(cudaFree(d_out_pos));



// const int numBits = 1;
// const int numBins = 1 << numBits;

// unsigned int *binHistogram = new unsigned int[numBins];
// unsigned int *binScan      = new unsigned int[numBins];

// unsigned int *vals_src =  new unsigned int[numElems * sizeof(unsigned int)];
// unsigned int *pos_src =  new unsigned int[numElems * sizeof(unsigned int)];
// memset(vals_src, 0 ,numElems * sizeof(unsigned int));
// memset(pos_src, 0 ,numElems * sizeof(unsigned int));


// checkCudaErrors(cudaMemcpy(vals_src, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
// checkCudaErrors(cudaMemcpy(pos_src, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));


// unsigned int *vals_dst =  new unsigned int[numElems * sizeof(unsigned int)];
// unsigned int *pos_dst =  new unsigned int[numElems * sizeof(unsigned int)];

// memset(vals_dst, 0 ,numElems * sizeof(unsigned int));
// memset(pos_dst, 0 ,numElems * sizeof(unsigned int));

// checkCudaErrors(cudaMemcpy(vals_dst, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
// checkCudaErrors(cudaMemcpy(pos_dst, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));


// //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
// for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
//   unsigned int mask = (numBins - 1) << i;

//   memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
//   memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

//   //perform histogram of data & mask into bins
//   for (unsigned int j = 0; j < numElems; ++j) {
//     unsigned int bin = (vals_src[j] & mask) >> i;
//     binHistogram[bin]++;
//   }

//   //perform exclusive prefix sum (scan) on binHistogram to get starting
//   //location for each bin
// //   for (unsigned int j = 1; j < numBins; ++j) {
// //     binScan[j] = binScan[j - 1] + binHistogram[j - 1];
// //   }
// binScan[1] = binScan[0] + binHistogram[0];

//   //Gather everything into the correct location
//   //need to move vals and positions
//   for (unsigned int j = 0; j < numElems; ++j) {
//     unsigned int bin = (vals_src[j] & mask) >> i;
//     vals_dst[binScan[bin]] = vals_src[j];
//     pos_dst[binScan[bin]]  = pos_src[j];
//     binScan[bin]++;
//   }

//   //swap the buffers (pointers only)
//   std::swap(vals_dst, vals_src);
//   std::swap(pos_dst, pos_src);
// }

// //we did an even number of iterations, need to copy from input buffer into output
// std::copy(vals_src, vals_src + numElems, vals_dst);
// std::copy(pos_src, pos_src + numElems, pos_dst);
//   checkCudaErrors(cudaMemcpy(d_outputVals, vals_dst, numElems*sizeof(unsigned),
//   cudaMemcpyHostToDevice));
//   checkCudaErrors(cudaMemcpy(d_outputPos, pos_dst, numElems*sizeof(unsigned),
//   cudaMemcpyHostToDevice));

// delete[] binHistogram;
// delete[] binScan;

}

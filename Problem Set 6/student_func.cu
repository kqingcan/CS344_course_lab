//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
 
__global__ void generate_mask_and_channel_init(
   const uchar4* d_sourceImg,
   const uchar4* d_destImg, 
   unsigned char * d_mask, 
   unsigned char * d_red_src,
   unsigned char * d_blue_src,
   unsigned char * d_green_src,
   unsigned char * d_red_dst,
   unsigned char * d_blue_dst,
   unsigned char * d_green_dst,
   unsigned int srcSize)
{
   int myId = blockDim.x * blockIdx.x + threadIdx.x;
   if(myId >= srcSize) return;
   d_mask[myId] = (d_sourceImg[myId].x + d_sourceImg[myId].y + d_sourceImg[myId].z < 3 * 255) ? 1 : 0;
   d_red_src[myId] = d_sourceImg[myId].x;
   d_blue_src[myId] = d_sourceImg[myId].y;
   d_green_src[myId] = d_sourceImg[myId].z;

   d_red_dst[myId] = d_destImg[myId].x;
   d_blue_dst[myId] = d_destImg[myId].y;
   d_green_dst[myId] = d_destImg[myId].z;
}

__global__ void distinct_pixel_kind(
   unsigned char * d_mask, 
   unsigned char * d_borderPixels, 
   unsigned char * d_strictInteriorPixels, 
   size_t numRowsSource, 
   size_t numColsSource, 
   unsigned int * d_interior_size)
{
   int myId = blockDim.x * blockIdx.x + threadIdx.x;
   if(myId >= numColsSource * numRowsSource) return;

   int row = myId / numColsSource;
   int col = myId  % numColsSource;
   if(row ==0 || col ==0 || row == numRowsSource -1 || col == numColsSource -1){
      return;
   }
   if(d_mask[myId]){
      if(d_mask[myId - 1] && 
         d_mask[myId + 1] && 
         d_mask[myId - numColsSource] &&
         d_mask[myId + numColsSource])
      {
         d_strictInteriorPixels[myId] = 1;
         d_borderPixels[myId] = 0; 
         atomicAdd(&d_interior_size[0],1); 
      }else{
         d_strictInteriorPixels[myId] = 0;
         d_borderPixels[myId] = 1;
      }
   }else{
      d_strictInteriorPixels[myId] = 0;
      d_borderPixels[myId] = 0;
   }
}

__global__ void compute_g(
   unsigned char * d_red_src, 
   unsigned char * d_blue_src, 
   unsigned char * d_green_src, 
   float * d_g_red, 
   float * d_g_blue, 
   float * d_g_green,
   unsigned char * d_strictInteriorPixels, 
   size_t numColsSource, 
   size_t numRowsSource)
{
   int myId = blockDim.x * blockIdx.x +  threadIdx.x;
   if(myId >= numColsSource * numRowsSource) return;
   if(d_strictInteriorPixels[myId] == 0) return;
   float r_sum = 4.f * d_red_src[myId];
   float b_sum = 4.f * d_blue_src[myId];
   float g_sum = 4.f * d_green_src[myId];

   r_sum -= (float)d_red_src[myId -1] + (float)d_red_src[myId +1];
   r_sum -= (float)d_red_src[myId - numColsSource] + (float)d_red_src[myId + numColsSource]; 
   b_sum -= (float)d_blue_src[myId -1] + (float)d_blue_src[myId +1];
   b_sum -= (float)d_blue_src[myId - numColsSource] + (float)d_blue_src[myId + numColsSource]; 
   g_sum -= (float)d_green_src[myId -1] + (float)d_green_src[myId +1];
   g_sum -= (float)d_green_src[myId - numColsSource] + (float)d_green_src[myId + numColsSource];
   
   d_g_red[myId] = r_sum;
   d_g_blue[myId] = b_sum;
   d_g_green[myId] = g_sum; 
}

__global__ void init_blended_vals(
   unsigned char * d_red_src, 
   unsigned char * d_blue_src, 
   unsigned char * d_green_src,
   float * d_blendedValsRed_1,
   float *d_blendedValsRed_2,
   float *d_blendedValsBlue_1,
   float *d_blendedValsBlue_2,
   float *d_blendedValsGreen_1,
   float *d_blendedValsGreen_2, 
   unsigned int srcSize)
{
   int myId = blockDim.x * blockIdx.x + threadIdx.x;
   if(myId >= srcSize) return;
   d_blendedValsRed_1[myId] = d_red_src[myId];
   d_blendedValsRed_2[myId] = d_red_src[myId];
   d_blendedValsBlue_1[myId] = d_blue_src[myId];
   d_blendedValsBlue_2[myId] = d_blue_src[myId];
   d_blendedValsGreen_1[myId] = d_green_src[myId];
   d_blendedValsGreen_2[myId] = d_green_src[myId];
}


__global__ void compute_iteration(
   unsigned char * d_destImg, 
   unsigned char * d_strictInteriorPixels,
   unsigned char *  d_borderPixels, 
   size_t numColsSource, 
   size_t numRowsSource,
   float * d_g, 
   float * f, 
   float* f_next)
{
   int myId = blockDim.x * blockIdx.x + threadIdx.x;
   if(myId >= numColsSource * numRowsSource) return;
   if(d_strictInteriorPixels[myId] == 0) return;

   float blendedSum = 0.f;
   float borderSum = 0.f;

   if(d_strictInteriorPixels[myId -1]){
      blendedSum +=f[myId - 1];
   }else{
      borderSum += d_destImg[myId -1];
   }

   if(d_strictInteriorPixels[myId + 1]){
      blendedSum += f[myId +1];
   }else{
      borderSum += d_destImg[myId + 1];
   }

   if(d_strictInteriorPixels[myId - numColsSource]){
      blendedSum += f[myId - numColsSource];
   }else{
      borderSum += d_destImg[myId - numColsSource];
   }

   if(d_strictInteriorPixels[myId + numColsSource]){
      blendedSum += f[myId + numColsSource];
   }else{
      borderSum +=d_destImg[myId + numColsSource];
   }

   float f_next_val = (blendedSum + borderSum + d_g[myId])/4.f;
   f_next[myId] = min(255.f, max(0.f,f_next_val));
}

__global__ void compute_iteration1(
    unsigned char * d_red_dst, 
    unsigned char * d_blue_dst, 
    unsigned char * d_green_dst, 
    unsigned char * d_strictInteriorPixels,
    unsigned char *  d_borderPixels, 
    size_t numColsSource, 
    size_t numRowsSource,
    float * d_g_red,
    float * d_g_blue,
    float * d_g_green, 
    float * f_r, 
    float* f_next_r,
    float * f_b, 
    float* f_next_b,
    float * f_g, 
    float* f_next_g)
 {
    int myId = blockDim.x * blockIdx.x + threadIdx.x;
    if(myId >= numColsSource * numRowsSource) return;
    if(d_strictInteriorPixels[myId] == 0) return;
 
    float blendedSum_r = 0.f;
    float blendedSum_b = 0.f;
    float blendedSum_g = 0.f;

    float borderSum_r = 0.f;
    float borderSum_b = 0.f;
    float borderSum_g = 0.f;
 
    if(d_strictInteriorPixels[myId -1]){
       blendedSum_r +=f_r[myId - 1];
       blendedSum_b +=f_b[myId - 1];
       blendedSum_g +=f_g[myId - 1];

    }else{
       borderSum_r += d_red_dst[myId -1];
       borderSum_b += d_blue_dst[myId -1];
       borderSum_g += d_green_dst[myId -1];
    }

    if(d_strictInteriorPixels[myId +1]){
        blendedSum_r +=f_r[myId + 1];
        blendedSum_b +=f_b[myId + 1];
        blendedSum_g +=f_g[myId + 1];
 
     }else{
        borderSum_r += d_red_dst[myId +1];
        borderSum_b += d_blue_dst[myId +1];
        borderSum_g += d_green_dst[myId +1];
     }

     if(d_strictInteriorPixels[myId - numColsSource]){
        blendedSum_r +=f_r[myId - numColsSource];
        blendedSum_b +=f_b[myId - numColsSource];
        blendedSum_g +=f_g[myId - numColsSource];
 
     }else{
        borderSum_r += d_red_dst[myId - numColsSource];
        borderSum_b += d_blue_dst[myId - numColsSource];
        borderSum_g += d_green_dst[myId - numColsSource];
     }
     if(d_strictInteriorPixels[myId + numColsSource]){
        blendedSum_r +=f_r[myId + numColsSource];
        blendedSum_b +=f_b[myId + numColsSource];
        blendedSum_g +=f_g[myId + numColsSource];
 
     }else{
        borderSum_r += d_red_dst[myId + numColsSource];
        borderSum_b += d_blue_dst[myId + numColsSource];
        borderSum_g += d_green_dst[myId + numColsSource];
     }
 
    float f_next_val_r = (blendedSum_r + borderSum_r + d_g_red[myId])/4.f;
    float f_next_val_b = (blendedSum_b + borderSum_b + d_g_blue[myId])/4.f;
    float f_next_val_g = (blendedSum_g + borderSum_g + d_g_green[myId])/4.f;

    f_next_r[myId] = min(255.f, max(0.f,f_next_val_r));
    f_next_b[myId] = min(255.f, max(0.f,f_next_val_b));
    f_next_g[myId] = min(255.f, max(0.f,f_next_val_g));

 }
 
__global__ void gather_result(
    uchar4 * d_blendedImg, 
    float * d_blendedValsRed, 
    float * d_blendedValsBlue, 
    float * d_blendedValsGreen,
    unsigned char * d_strictInteriorPixels, 
    unsigned int srcSize)
{
    int myId = blockDim.x *blockIdx.x + threadIdx.x;

    if(myId >= srcSize) return;
    if(d_strictInteriorPixels[myId] ==0) return;
    d_blendedImg[myId].x = d_blendedValsRed[myId];
    d_blendedImg[myId].y = d_blendedValsBlue[myId];
    d_blendedImg[myId].z = d_blendedValsGreen[myId];
}

void your_blend(const uchar4* const h_sourceImg,  //IN
   const size_t numRowsSource, const size_t numColsSource,
   const uchar4* const h_destImg, //IN
   uchar4* const h_blendedImg) //OUT
{
/* To Recap here are the steps you need to implement
1) Compute a mask of the pixels from the source image to be copied
The pixels that shouldn't be copied are completely white, they
have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

2) Compute the interior and border regions of the mask.  An interior
pixel has all 4 neighbors also inside the mask.  A border pixel is
in the mask itself, but has at least one neighbor that isn't.

3) Separate out the incoming image into three separate channels

4) Create two float(!) buffers for each color channel that will
act as our guesses.  Initialize them to the respective color
channel of the source image since that will act as our intial guess.

5) For each color channel perform the Jacobi iteration described 
above 800 times.

6) Create the output image by replacing all the interior pixels
in the destination image with the result of the Jacobi iterations.
Just cast the floating point values to unsigned chars since we have
already made sure to clamp them to the correct range.

Since this is final assignment we provide little boilerplate code to
help you.  Notice that all the input/output pointers are HOST pointers.

You will have to allocate all of your own GPU memory and perform your own
memcopies to get data in and out of the GPU memory.

Remember to wrap all of your calls with checkCudaErrors() to catch any
thing that might go wrong.  After each kernel call do:

cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

to catch any errors that happened while executing the kernel.
*/
  
   unsigned int srcSize = numRowsSource * numColsSource;
   dim3 blockSize(128, 1, 1);
   dim3 gridSize((srcSize + blockSize.x -1)/ blockSize.x , 1, 1);
   unsigned char * d_mask ;

   uchar4 * d_sourceImg;
   uchar4 * d_destImg;
   unsigned char * d_red_src;
   unsigned char * d_blue_src;
   unsigned char * d_green_src;
   unsigned char * d_red_dst;
   unsigned char * d_blue_dst;
   unsigned char * d_green_dst;

   checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * srcSize));
   checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * srcSize));
   checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));

   checkCudaErrors(cudaMalloc(&d_red_src, sizeof(unsigned char) * srcSize ));
   checkCudaErrors(cudaMalloc(&d_blue_src, sizeof(unsigned char) * srcSize ));
   checkCudaErrors(cudaMalloc(&d_green_src, sizeof(unsigned char) * srcSize ));
   checkCudaErrors(cudaMalloc(&d_red_dst, sizeof(unsigned char) * srcSize ));
   checkCudaErrors(cudaMalloc(&d_blue_dst, sizeof(unsigned char) * srcSize ));
   checkCudaErrors(cudaMalloc(&d_green_dst, sizeof(unsigned char) * srcSize ));


   checkCudaErrors(cudaMalloc(&d_mask, srcSize * sizeof(unsigned char)));
   checkCudaErrors(cudaMemset(d_mask, 0 , srcSize * sizeof(unsigned char)));

   generate_mask_and_channel_init<<<gridSize, blockSize>>>(d_sourceImg, d_destImg ,d_mask,
       d_red_src, d_blue_src, d_green_src,
       d_red_dst, d_blue_dst, d_green_dst,srcSize);
   
    
   unsigned char * d_borderPixels;
   unsigned char * d_strictInteriorPixels;
   unsigned int * d_interior_size;
   unsigned int interior_size = 0;
   checkCudaErrors(cudaMalloc(&d_interior_size, sizeof(unsigned int)));
   checkCudaErrors(cudaMalloc(&d_borderPixels, sizeof(unsigned char) * srcSize));
   checkCudaErrors(cudaMalloc(&d_strictInteriorPixels, sizeof(unsigned char) * srcSize));
   
   checkCudaErrors(cudaMemset(d_interior_size, 0 , sizeof(unsigned int)));
   checkCudaErrors(cudaMemset(d_borderPixels, 0 ,sizeof(unsigned char) * srcSize));
   checkCudaErrors(cudaMemset(d_strictInteriorPixels, 0 ,sizeof(unsigned char) * srcSize));
   distinct_pixel_kind<<<gridSize, blockSize>>>(d_mask, d_borderPixels, d_strictInteriorPixels,
   numRowsSource, numColsSource, d_interior_size);
   checkCudaErrors(cudaMemcpy(&interior_size, d_interior_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
   float *g_red   = new float[srcSize];
   float *g_blue  = new float[srcSize];
   float *g_green = new float[srcSize];
   float * d_g_red;
   float * d_g_blue;
   float * d_g_green;
   checkCudaErrors(cudaMalloc(&d_g_red, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_g_blue, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_g_green, sizeof(float) * srcSize));

   checkCudaErrors(cudaMemset(d_g_red, 0 , sizeof(float)* srcSize));
   checkCudaErrors(cudaMemset(d_g_blue, 0 , sizeof(float)* srcSize));
   checkCudaErrors(cudaMemset(d_g_green, 0 , sizeof(float)* srcSize));

   compute_g<<<gridSize, blockSize>>>(d_red_src, d_blue_src, d_green_src,
      d_g_red, d_g_blue, d_g_green,d_strictInteriorPixels, numColsSource, numRowsSource);
   
   
   //for each color channel we'll need two buffers and we'll ping-pong between them
   float * d_blendedValsRed_1;
   float *d_blendedValsRed_2;
   
   float *d_blendedValsBlue_1 ;
   float *d_blendedValsBlue_2 ;
   
   float *d_blendedValsGreen_1;
   float *d_blendedValsGreen_2 ;
   checkCudaErrors(cudaMalloc(&d_blendedValsRed_1, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_blendedValsRed_2, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_blendedValsBlue_1, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_blendedValsBlue_2, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_blendedValsGreen_1, sizeof(float) * srcSize));
   checkCudaErrors(cudaMalloc(&d_blendedValsGreen_2, sizeof(float) * srcSize));

   init_blended_vals<<<gridSize, blockSize>>>(d_red_src, d_blue_src, d_green_src, 
    d_blendedValsRed_1, d_blendedValsRed_2, 
    d_blendedValsBlue_1, d_blendedValsBlue_2, d_blendedValsGreen_1, d_blendedValsGreen_2, srcSize);

   
   //Perform the solve on each color channel
   const size_t numIterations = 800;
   for (size_t i = 0; i < numIterations; ++i) {
    //    compute_iteration<<<gridSize, blockSize>>>(d_red_dst, d_strictInteriorPixels,d_borderPixels,numColsSource, numRowsSource, d_g_red, d_blendedValsRed_1, d_blendedValsRed_2);
    //    compute_iteration<<<gridSize, blockSize>>>(d_blue_dst, d_strictInteriorPixels,d_borderPixels,numColsSource, numRowsSource, d_g_blue, d_blendedValsBlue_1, d_blendedValsBlue_2);
    //    compute_iteration<<<gridSize, blockSize>>>(d_green_dst, d_strictInteriorPixels,d_borderPixels,numColsSource, numRowsSource, d_g_green, d_blendedValsGreen_1, d_blendedValsGreen_2);
    compute_iteration1<<<gridSize, blockSize>>>(d_red_dst,d_blue_dst,d_green_dst,
        d_strictInteriorPixels,d_borderPixels,numColsSource, numRowsSource,
        d_g_red,d_g_blue,d_g_green,
        d_blendedValsRed_1, d_blendedValsRed_2, d_blendedValsBlue_1,d_blendedValsBlue_2,d_blendedValsGreen_1, d_blendedValsGreen_2);

       std::swap(d_blendedValsRed_1, d_blendedValsRed_2);
       std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);
       std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2);
   }
   
   std::swap(d_blendedValsRed_1,   d_blendedValsRed_2);   //put output into _2
   std::swap(d_blendedValsBlue_1,  d_blendedValsBlue_2);  //put output into _2
   std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2); //put output into _2
   
   //copy the destination image to the output
   memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);
   uchar4 * d_blendedImg;
   checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * srcSize));
   checkCudaErrors(cudaMemcpy(d_blendedImg, h_blendedImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice));

   
   gather_result<<<gridSize, blockSize>>>(d_blendedImg, d_blendedValsRed_2, 
    d_blendedValsBlue_2, d_blendedValsGreen_2, d_strictInteriorPixels, srcSize);
 
   checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4)* srcSize, cudaMemcpyDeviceToHost));
   
   checkCudaErrors(cudaFree(d_mask));
   checkCudaErrors(cudaFree(d_sourceImg));
   checkCudaErrors(cudaFree(d_destImg));
   checkCudaErrors(cudaFree(d_red_src));
   checkCudaErrors(cudaFree(d_red_dst));
   checkCudaErrors(cudaFree(d_blue_src));
   checkCudaErrors(cudaFree(d_blue_dst));
   checkCudaErrors(cudaFree(d_green_src));
   checkCudaErrors(cudaFree(d_green_dst));
   checkCudaErrors(cudaFree(d_borderPixels));
   checkCudaErrors(cudaFree(d_strictInteriorPixels));
   checkCudaErrors(cudaFree(d_interior_size));
   checkCudaErrors(cudaFree(d_g_red));
   checkCudaErrors(cudaFree(d_g_blue));
   checkCudaErrors(cudaFree(d_g_green));
   checkCudaErrors(cudaFree(d_blendedValsRed_1));
   checkCudaErrors(cudaFree(d_blendedValsRed_2));
   checkCudaErrors(cudaFree(d_blendedValsBlue_1));
   checkCudaErrors(cudaFree(d_blendedValsBlue_2));
   checkCudaErrors(cudaFree(d_blendedValsGreen_1));
   checkCudaErrors(cudaFree(d_blendedValsGreen_2));
   checkCudaErrors(cudaFree(d_blendedImg));


}
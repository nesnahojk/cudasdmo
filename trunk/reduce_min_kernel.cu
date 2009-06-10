#include "sharedmem.cuh"
#include <iostream>
#include "cublas.h"
#include <float.h>
#include "reduce_min_kernel.h"

template <class T>
__host__ void min_with_index_host(T x1, T x2,int y1, int y2,T &out_val, int & out_index)
{



       if(x1==fminf(x1,x2))
	{
	  out_val=x1;
	  out_index=y1;
	}
      else
	{
	  out_val=x2;
	  out_index=y2;
	}
}

template <class T>
__device__ void min_with_index(T x1, T x2,int y1, int y2,T &out_val, int & out_index)
{



       if(x1==fminf(x1,x2))
	{
	  out_val=x1;
	  out_index=y1;
	}
      else
	{
	  out_val=x2;
	  out_index=y2;
	}
}




template <class T, unsigned int blockSize>
__global__ void reduce_min_kernel(T *g_odata, T *g_idata,int* index_o, unsigned int n)
{
   __shared__ T sdata[blockSize];
   __shared__ int index[blockSize];
   

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
   unsigned int gridSize = blockSize*2*gridDim.x;


   //   T thMin = fminf(g_idata[i], g_idata[i + blockSize]);
   T thMin=0;
   int thMin_index=0;
   min_with_index<T>(g_idata[i],g_idata[i+blockSize],i,i+blockSize,thMin,thMin_index);
   i += gridSize;
   while (i < n)
   {
     T a=0;
     int a_index;

         min_with_index<T>(g_idata[i],g_idata[i+blockSize],i,i+blockSize,a,a_index);
     // = fminf(g_idata[i], g_idata[i + blockSize]);

         min_with_index<T>(thMin,a,thMin_index,a_index,thMin,thMin_index);
     //     thMin = fminf(thMin, a);
     i += gridSize;
   }
   sdata[tid] =thMin;
   index[tid]=thMin_index;
   __syncthreads();


      if (blockSize >= 512) { if (tid < 256) { min_with_index<T>(sdata[tid],sdata[tid+256],index[tid],index[tid+256],sdata[tid],index[tid]); } __syncthreads(); }
      if (blockSize >= 256) { if (tid < 128) { min_with_index<T>(sdata[tid],sdata[tid+128],index[tid],index[tid+128],sdata[tid],index[tid]); } __syncthreads(); }
      if (blockSize >= 128) { if (tid <  64) { min_with_index<T>(sdata[tid],sdata[tid+64],index[tid],index[tid+64],sdata[tid],index[tid]); } __syncthreads(); }
   
   if (tid < 32)
   {
       if (blockSize >=  64) { min_with_index<T>(sdata[tid],sdata[tid+32],index[tid],index[tid+32],sdata[tid],index[tid]); }
       if (blockSize >=  32) { min_with_index<T>(sdata[tid],sdata[tid+16],index[tid],index[tid+16],sdata[tid],index[tid]); }
       if (blockSize >=  16) { min_with_index<T>(sdata[tid],sdata[tid+8],index[tid],index[tid+8],sdata[tid],index[tid]); }
       if (blockSize >=   8) { min_with_index<T>(sdata[tid],sdata[tid+4],index[tid],index[tid+4],sdata[tid],index[tid]); }
       if (blockSize >=   4) { min_with_index<T>(sdata[tid],sdata[tid+2],index[tid],index[tid+2],sdata[tid],index[tid]); }
       if (blockSize >=   2) { min_with_index<T>(sdata[tid],sdata[tid+1],index[tid],index[tid+1],sdata[tid],index[tid]); }
   }
   
   // write result for this block to global mem
   if (tid == 0)
     {
       g_odata[blockIdx.x] = sdata[0];
       index_o[blockIdx.x]=index[0];
     }

}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduce_min(T *d_idata, int size, int & final_index)
{
 const int maxThreads = 128;
 const int maxBlocks  = 128;
 int threads = 1;

  if(size%2!=0)
    {
      size=size-1;
    }

 if( size != 1 ) {
   threads = (size < maxThreads*2) ? size / 2 : maxThreads;
 }
 int blocks = size / (threads * 2);
 blocks = min(maxBlocks, blocks);


  T * d_odata;
  cublasAlloc(blocks, sizeof(T), (void**)&d_odata);

  int * index_o;
  cublasAlloc(blocks, sizeof(int), (void**)&index_o);


 dim3 dimBlock(threads, 1, 1);
 dim3 dimGrid(blocks, 1, 1);
 int smemSize = threads * sizeof(T);

 switch (threads)
 {
 case 512:
   reduce_min_kernel<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case 256:
   reduce_min_kernel<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case 128:
   reduce_min_kernel<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case 64:
   reduce_min_kernel<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case 32:
   reduce_min_kernel<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case 16:
   reduce_min_kernel<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case  8:
   reduce_min_kernel<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case  4:
   reduce_min_kernel<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case  2:
   reduce_min_kernel<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 case  1:
   reduce_min_kernel<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata,index_o, size); break;
 default:
     exit(1);
 }

 T* h_odata = new T[blocks];
 int * h_index=new int[blocks];
 cudaMemcpy( h_index, index_o, blocks*sizeof(int), cudaMemcpyDeviceToHost);
 cudaMemcpy( h_odata, d_odata, blocks*sizeof(T), cudaMemcpyDeviceToHost);

 T result = h_odata[0];
 int result_index=h_index[0];
 for( int i = 1; i < blocks; i++ ) {
      min_with_index_host<T>(result,h_odata[i],result_index,h_index[i],result,result_index);
 }
 delete[] h_odata;
 final_index=result_index;
 return result;
}




template <class T>
int IndexOfMin(T * didata, int N)
{
  int final_index=0;
  bool odd=false;
  if(N%2!=0)
    {
      N=N-1;
      odd=true;
    }

  float value=reduce_min(didata,N,final_index);
  
  if(odd)
    {
      T * single=new T[1];
      //      ublasGetVector (int n, int elemSize, const void *x,
      //        int incx, void *y, int incy)
      cublasGetVector(1,sizeof(T),&didata[N],1,single,1);
      if(single[0]<value)
	{
	  final_index=N;
	}
    }

  return final_index;
}

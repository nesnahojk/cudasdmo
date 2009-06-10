#ifndef __REDUCTION_MIN__
#define __REDUCTION_MIN__

template <class T, unsigned int blockSize>
void reduce_min_kernel(T *g_odata, T *g_idata,int* index_o, unsigned int n);

template <class T>
void min_with_index_host(T x1, T x2,int y1, int y2,T &out_val, int & out_index);

template <class T>
void min_with_index(T x1, T x2,int y1, int y2,T &out_val, int & out_index);

template <class T>
T reduce_min(T *d_idata, int size, int & final_index);

template <class T>
int IndexOfMin(T * didata, int N);



#endif



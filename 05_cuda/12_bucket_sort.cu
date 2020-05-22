#include <cstdio>
/* #include <cstdlib> */
/* #include <vector> */

__global__ void sort(int *key, int *bucket, int n, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  extern __shared__ int b[];                           // buckets shared in each block
  for (int j=0; j<range; j++) b[j] = 0;
  atomicAdd(b+key[i], 1);                              // counting in each block
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int j=0; j<range; j++) {
      atomicAdd(bucket+j, b[j]);                       // collecting to global mem
    }
  }
  __syncthreads();
  int acc = bucket[0];
  int j = 0;
  while (1) {                                          // reallocating
    if (i < acc) {
      key[i] = j;
      break;
    } else {
      acc += bucket[++j];
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  /* std::vector<int> key(n); */
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int M = 32;
  int *bucket;                                         // global mem buckets
  cudaMallocManaged(&bucket, range*sizeof(int));
  sort<<<(n+M-1)/M,M,range>>>(key, bucket, n, range);
  cudaDeviceSynchronize();

  /* std::vector<int> bucket(range); */ 
  /* for (int i=0; i<range; i++) { */
  /*   bucket[i] = 0; */
  /* } */
  /* for (int i=0; i<n; i++) { */
  /*   bucket[key[i]]++; */
  /* } */
  /* for (int i=0, j=0; i<range; i++) { */
  /*   for (; bucket[i]>0; bucket[i]--) { */
  /*     key[j++] = i; */
  /*   } */
  /* } */

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

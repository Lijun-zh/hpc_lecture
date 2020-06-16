#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;

__global__ void build_up_b(int nx, int ny,
    float *u, float *v, float *b,
    float dt, float dx, float dy, float rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nx) return;
  if(i%nx == nx-1) return;
  if(i%nx == 0) return;
  if(i >= nx*(ny-1)) return;
      b[i] = rho * (1 / dt *
            ((u[i+1] - u[i-1]) /
             (2. * dx) + (v[i+nx] - v[i-nx]) / (2. * dy)) -
            pow((u[i+1] - u[i-1]) / (2. * dx), 2) -
              2. * ((u[i+nx] - u[i-nx]) / (2. * dy) *
                   (v[i+1] - v[i-1]) / (2. * dx)) -
                   pow((v[i+nx] - v[i-nx]) / (2. * dy), 2));
}

__global__ void pressure_poisson(int nx, int ny,
    float *p, float *pn, float *b,
    float dx, float dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if((i >= nx) && (i%nx != nx-1) &&
     (i%nx != 0) && (i < nx*(ny-1))) {
    p[i] =  ((pn[i+1] + pn[i-1]) * dy*dy +
                (pn[i+nx] + pn[i-nx]) * dx*dx) /
                (2 * (dx*dx + dy*dy)) -
                dx*dx * dy*dy / (2 *(dx*dx + dy*dy)) *
                b[i]; 
  }
}
__global__ void pressure_poisson_bc(int nx, int ny, float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nx*ny) return;
  if(i%nx == nx-1) p[i] = p[i-1];
  else if(i < nx) p[i] = p[i+nx];
  else if(i%nx == 0) p[i] = p[i+1];
  else if(i >= nx*(ny-1)) p[i] = 0;
}

__global__ void calc_uv(int nx, int ny,
    float *u, float*v, float *un, float *vn, float *p,
    float dt, float dx, float dy, float rho, float nu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<nx*ny) {
    if((i >= nx) && (i%nx != nx-1) &&
       (i%nx != 0) && (i < nx*(ny-1))) {
      u[i] =  un[i]-
                 un[i] * dt / dx *
                (un[i] - un[i-1]) -
                 vn[i] * dt / dy *
                (un[i] - un[i-nx]) -
                 dt / (2. * rho * dx) * (p[i+1] - p[i-1]) +
                 nu * (dt / (dx*dx) *
                (un[i+1] - 2. * un[i] + un[i-1]) +
                 dt / (dy*dy) *
                (un[i+nx] - 2. * un[i] + un[i-nx]));
    }
  } else {
    i -= nx*ny;
    if((i >= nx) && (i%nx != nx-1) &&
       (i%nx != 0) && (i < nx*(ny-1))) {
      v[i] = vn[i] -
                un[i] * dt / dx *
               (vn[i] - vn[i-1]) -
                vn[i] * dt / dy *
               (vn[i] - vn[i-nx]) -
                dt / (2. * rho * dy) * (p[i+nx] - p[i-nx]) +
                nu * (dt / (dx*dx) *
               (vn[i+1] - 2. * vn[i] + vn[i-1]) +
                dt / (dy*dy) *
               (vn[i+nx] - 2. * vn[i] + vn[i-nx]));
    }
  }
}
__global__ void calc_uv_bc(int nx, int ny, float *u, float *v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nx*ny) { 
    if(i%nx == nx-1) u[i] = 0;
    else if(i < nx) u[i] = 0;
    else if(i%nx == 0) u[i] = 0;
    else if(i >= nx*(ny-1)) u[i] = 1;
  } else {
    i -= nx*ny;
    if(i >= nx*ny) return;
    if(i%nx == nx-1) v[i] = 0;
    else if(i < nx) v[i] = 0;
    else if(i%nx == 0) v[i] = 0;
    else if(i >= nx*(ny-1)) v[i] = 0;

  }
}

__device__ __managed__ float usum;
__device__ __managed__ float udiff;

__device__ float warpSum(float a) {
  for (int offset=16; offset>0; offset >>= 1)
    a += __shfl_down_sync(0xffffffff, a, offset);
  return a;
}

__global__ void l1diff(int n, float *a, float *an, float &diff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  float b = warpSum(abs(a[i]-an[i]));
  if ((threadIdx.x & 31) == 0)
    atomicAdd(&diff, b);
}

__global__ void l1sum(int n, float *a, float &sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=n) return;
  float b = warpSum(abs(a[i])); 
  if ((threadIdx.x & 31) == 0)
    atomicAdd(&sum, b);
}

int cavity_flow(float eps, int tb, int nx, int ny,
    float *u, float *un, float *v, float *vn,
    float *p, float *pn, float *b,
    float dt, float dx, float dy, float rho, float nu) {

  int nb = (nx*ny+tb-1)/tb;    // number of blocks
  int nt = 0;                  // number of timesteps
  udiff = 233;
  for(;udiff > eps; nt++) {
    cudaMemcpy(un, u, nx*ny*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(vn, v, nx*ny*sizeof(float), cudaMemcpyDefault);

    build_up_b<<<nb,tb>>>
      (nx, ny, u, v, b, dt, dx, dy, rho);
    for (int nit=0; nit<50; nit++) {
      cudaMemcpy(pn, p, nx*ny*sizeof(float), cudaMemcpyDefault);
      pressure_poisson<<<nb,tb>>>(nx, ny, p, pn, b, dx, dy); 
      pressure_poisson_bc<<<nb,tb>>>(nx, ny, p); 
    }
    calc_uv<<<nb*2,tb>>>(nx, ny, u, v, un, vn, p, dt, dx, dy, rho, nu);
    calc_uv_bc<<<nb*2,tb>>>(nx, ny, u, v);

    udiff = 0;
    usum = 0;
    l1diff<<<nb,tb>>>(nx*ny, u, un, udiff);
    l1sum<<<nb,tb>>>(nx*ny, u, usum);
    cudaDeviceSynchronize();
    udiff /= usum;
  }
  return nt;
}


int main() {
  const int nx = 41, ny = 41;
  const int tb = 1024;     // number of threads per block
  float *u, *v, *p, *un, *vn, *pn, *b;
  int size = nx*ny*sizeof(float);
  cudaMallocManaged(&u, size);
  cudaMallocManaged(&v, size);
  cudaMallocManaged(&p, size);
  cudaMallocManaged(&un, size);
  cudaMallocManaged(&vn, size);
  cudaMallocManaged(&pn, size);
  cudaMallocManaged(&b, size);
  cudaMemset(u, 0, size); 
  cudaMemset(v, 0, size); 
  cudaMemset(p, 0, size); 
  cudaMemset(un, 0, size); 
  cudaMemset(vn, 0, size); 
  cudaMemset(pn, 0, size); 
  cudaMemset(b, 0, size); 
  float dt = .001;
  float dx = 2. / (nx - 1.);
  float dy = 2. / (ny - 1.);
  float rho = 1., nu = .1;
  float eps = .0000001;

  int nt;
  auto tic = chrono::steady_clock::now();
  nt = cavity_flow
    (eps, tb, nx, ny, u, un, v, vn, p, pn, b, dt, dx, dy, rho, nu);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("Reaching convergence after %d steps.\n", nt);
  printf("Elapsed time is %lf s.\n",time);

  float ures = 0, vres = 0, pres = 0;
  for(int i=0; i<ny*nx; i++) {
    ures += abs(u[i]);
    vres += abs(v[i]);
    pres += abs(p[i]);
  }
  printf("Sum(|u|)=%f\n", ures);
  printf("Sum(|v|)=%f\n", vres);
  printf("Sum(|p|)=%f\n", pres);

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
  cudaFree(b);

  return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xav = _mm256_load_ps(x);
  __m256 yav = _mm256_load_ps(y);
  __m256 mav = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    float mask[N], temp[N];
    for(int j=0; j<N; j++) mask[j] = 1; mask[i] = 0;
    __m256 maskv = _mm256_load_ps(mask);
    maskv = _mm256_cmp_ps(maskv, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 xiv = _mm256_set1_ps(x[i]);
    __m256 yiv = _mm256_set1_ps(y[i]);

    __m256 rxv = _mm256_sub_ps(xiv, xav);
    __m256 rx2v = _mm256_mul_ps(rxv, rxv);
    __m256 ryv = _mm256_sub_ps(yiv, yav);
    __m256 ry2v = _mm256_mul_ps(ryv, ryv);
    __m256 rrv = _mm256_rsqrt_ps(_mm256_add_ps(rx2v, ry2v));
    rrv = _mm256_blendv_ps(_mm256_setzero_ps(), rrv, maskv);

    __m256 fxiv = _mm256_mul_ps(rxv, mav);
    fxiv = _mm256_mul_ps(fxiv, rrv);
    fxiv = _mm256_mul_ps(fxiv, rrv);
    fxiv = _mm256_mul_ps(fxiv, rrv);
    __m256 pfxiv = _mm256_permute2f128_ps(fxiv, fxiv, 1);
    fxiv = _mm256_add_ps(fxiv, pfxiv);
    fxiv = _mm256_hadd_ps(fxiv, fxiv);
    fxiv = _mm256_hadd_ps(fxiv, fxiv);
    _mm256_store_ps(temp, fxiv);
    fx[i] = -temp[0];
    __m256 fyiv = _mm256_mul_ps(ryv, mav);
    fyiv = _mm256_mul_ps(fyiv, rrv);
    fyiv = _mm256_mul_ps(fyiv, rrv);
    fyiv = _mm256_mul_ps(fyiv, rrv);
    __m256 pfyiv = _mm256_permute2f128_ps(fyiv, fyiv, 1);
    fyiv = _mm256_add_ps(fyiv, pfyiv);
    fyiv = _mm256_hadd_ps(fyiv, fyiv);
    fyiv = _mm256_hadd_ps(fyiv, fyiv);
    _mm256_store_ps(temp, fyiv);
    fy[i] = -temp[0];
    /* for(int j=0; j<N; j++) { */
    /*   if(i != j) { */
    /*     float rx = x[i] - x[j]; */
    /*     float ry = y[i] - y[j]; */
    /*     float r = std::sqrt(rx * rx + ry * ry); */
    /*     fx[i] -= rx * m[j] / (r * r * r); */
    /*     fy[i] -= ry * m[j] / (r * r * r); */
    /*   } */
    /* } */
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

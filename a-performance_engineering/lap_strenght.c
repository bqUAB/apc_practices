#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

float stencil ( float v1, float v2, float v3, float v4)
{
  return (v1 + v2 + v3 + v4) * 0.25f;
}

float max_error ( float prev_error, float old, float new )
{
  float t= sqrtf( fabsf( new - old ) );
  if (t> prev_error)
    return t;
  return prev_error;
}

void laplace_step(float *in, float *out, int n)
{
  int i, j;
  for ( i=1; i < n-1; i++ )
    for ( j=1; j < n-1; j++ )
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
}

float laplace_error (float *old, float *new, int n)
{
  int i, j;
  float error=0.0f;
  for ( i=1; i < n-1; i++ )
    for ( j=1; j < n-1; j++ )
      error = max_error( error, old[j*n+i], new[j*n+i] );
  return error;
}

void laplace_copy(float *in, float *out, int n)
{
  int i, j;
  for ( i=1; i < n-1; i++ )
    for ( j=1; j < n-1; j++ )
      out[j*n+i]= in[j*n+i];
}


void laplace_init(float *in, int n)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*n*sizeof(float));
  for (i=0; i<n; i++)  in[    i    ] = 0.f;
  for (i=0; i<n; i++)  in[(n-1)*n+i] = 0.f;
  for (i=0; i<n; i++)  in[   i*n   ] = sinf(pi*i / (n-1));
  for (i=0; i<n; i++)  in[ i*n+n-1 ] = sinf(pi*i / (n-1))*expf(-pi);
}

int main(int argc, char** argv)
{
  int n = 4096;
  int iter_max = 1000;
  float *A, *temp;
    
  const float tol = 1.0e-3f;
  float error= 1.0f;    

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  A    = (float*) malloc( n*n*sizeof(float) );
  temp = (float*) malloc( n*n*sizeof(float) );

  //  set boundary conditions
  laplace_init (A, n);
  laplace_init (temp, n);

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );

  int iter = 0;
  while ( error > tol && iter < iter_max )
  {
    iter++;
    laplace_step (A, temp, n);
    error= laplace_error (A, temp, n);
    laplace_copy (temp, A, n);
  }
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);
}

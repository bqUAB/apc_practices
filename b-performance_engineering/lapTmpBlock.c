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
  float t= fabsf( new - old );
  return t> prev_error? t: prev_error;
}

float step_error_line(float error, float *inUp, float *in, float *inDown, float *out, int n)
{
  int i;
  out[0]=in[0];
  #pragma omp simd reduction(max:error)
  for ( i=1; i < n-1; i++ )
  {
    out[i]=  stencil ( inUp[i], in[i-1], in[i+1], inDown[i] );
    error = max_error( error, out[i], in[i] );
  }
  out[n-1]= in[n-1];
  return error;
}

void laplace_init(float *in, int n)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*n*sizeof(float));
  for (i=0; i<n; i++) {
    float V = in[i*n] = sinf(pi*i / (n-1));
    in[ i*n+n-1 ] = V*expf(-pi);
  }
}

#define ROW(M,x)  ((M)+(x)*n)

int main(int argc, char** argv)
{
  int n = 4096, iter_max = 1000;
  float *A, *t, err= 1.0f;    
    
  const float tol = 1.0e-5f;

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  if (iter_max & 1) // n is odd
  { 
    printf("Maximum number of iterations must be even\n" );
    exit(1);
  }

  A = (float*) malloc( n*n*sizeof(float) );
  t = (float*) malloc( 3*n*sizeof(float) );

  //  set boundary conditions: Only in A[]
  laplace_init (A, n);
  A[(n/128)*n+n/128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );

  int iter = 0;
  while ( err > tol*tol && iter < iter_max )
  {
    int j;  iter+=2;
    // Prologue
         step_error_line(0.0f, ROW(A,0), ROW(A,1), ROW(A,2), ROW(t,1), n);
         step_error_line(0.0f, ROW(A,1), ROW(A,2), ROW(A,3), ROW(t,2), n);
    err= step_error_line(0.0f, ROW(A,0), ROW(t,1), ROW(t,2), ROW(A,1), n);
    // Main Loop
    for ( j=3; j < n-1; j++ ) 
    {
           step_error_line(err, ROW(A,j-1),     ROW(A,j),       ROW(A,j+1), ROW(t,j%3), n);
      err= step_error_line(err, ROW(t,(j-2)%3), ROW(t,(j-1)%3), ROW(t,j%3), ROW(A,j-1), n);
    }
    // Epilogue
    err= step_error_line(err, ROW(t,(n-3)%3), ROW(t,(n-2)%3), ROW(A,n-1), ROW(A,n-2), n);
  }
  err = sqrtf( err );

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, err);
  printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);

  free(A);  free(t);
}

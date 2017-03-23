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

float lap_step_line(float error, float *inUp, float *in, float *inDown, float *out, int n)
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

#define FILA(M,x)  ((M)+(x)*n)

int main(int argc, char** argv)
{
  int    n = 4096, iter_max = 1000;
  float *A, *t;
    
  const float tol = 1.0e-5f;
  float err= 1.0f;    

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  if (iter_max & 3) // is not multiple of 4
  { 
    printf("Maximum number of iterations must be multiple of 4\n" );
    exit(1);
  }

  A = (float*) malloc( n*n*sizeof(float) );
  t = (float*) malloc( 6*n*sizeof(float) );

  //  set boundary conditions: Only in A[]
  laplace_init (A, n);
  A[(n/128)*n+n/128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );

  int iter = 0;
  while ( err > tol*tol && iter < iter_max )
  {
    int j;
    iter+=4;

    // Prologue
         lap_step_line ( 0.0, FILA(A,0), FILA(A,1), FILA(A,2), FILA(t,1), n );

         lap_step_line ( 0.0, FILA(A,1), FILA(A,2), FILA(A,3), FILA(t,2), n );
         lap_step_line ( 0.0, FILA(A,0), FILA(t,1), FILA(t,2), FILA(A,1), n );

         lap_step_line ( 0.0, FILA(A,2), FILA(A,3), FILA(A,4), FILA(t,3), n );
         lap_step_line ( 0.0, FILA(t,1), FILA(t,2), FILA(t,3), FILA(A,2), n );

         lap_step_line ( 0.0, FILA(A,3), FILA(A,4), FILA(A,5), FILA(t,4), n );
         lap_step_line ( 0.0, FILA(t,2), FILA(t,3), FILA(t,4), FILA(A,3), n );
         lap_step_line ( 0.0, FILA(A,0), FILA(A,1), FILA(A,2), FILA(t,1), n );


         lap_step_line ( 0.0, FILA(A,4), FILA(A,5), FILA(A,6), FILA(t,5), n );
         lap_step_line ( 0.0, FILA(t,3), FILA(t,4), FILA(t,5), FILA(A,4), n );
         lap_step_line ( 0.0, FILA(A,1), FILA(A,2), FILA(A,3), FILA(t,2), n );
    err= lap_step_line ( 0.0, FILA(A,0), FILA(t,1), FILA(t,2), FILA(A,1), n );

    // Main Loop
    for ( j=6; j < n-1; j++ ) 
    {
           lap_step_line ( 0.0, FILA(A,j-1),     FILA(A,j),       FILA(A,j+1),     FILA(t,j%6),     n);
           lap_step_line ( 0.0, FILA(t,(j-2)%6), FILA(t,(j-1)%6), FILA(t,j%6),     FILA(A,j-1),     n);
           lap_step_line ( 0.0, FILA(A,j-4),     FILA(A,j-3),     FILA(A,j-2),     FILA(t,(j-3)%6), n);
      err= lap_step_line ( err, FILA(t,(j-5)%6), FILA(t,(j-4)%6), FILA(t,(j-3)%6), FILA(A,j-4),     n);
    }

    // Epilogue
         lap_step_line ( 0.0, FILA(t,(n-3)%6), FILA(t,(n-2)%6), FILA(A,n-1),     FILA(A,n-2),     n);
         lap_step_line ( 0.0, FILA(A,n-5),     FILA(A,n-4),     FILA(A,n-3),     FILA(t,(n-4)%6), n);
    err= lap_step_line ( err, FILA(t,(n-6)%6), FILA(t,(n-5)%6), FILA(t,(n-4)%6), FILA(A,n-5),     n);

         lap_step_line ( 0.0, FILA(A,n-4),     FILA(A,n-3),     FILA(A,n-2),     FILA(t,(n-3)%6), n);
    err= lap_step_line ( err, FILA(t,(n-5)%6), FILA(t,(n-4)%6), FILA(t,(n-3)%6), FILA(A,n-4),     n);

         lap_step_line ( 0.0, FILA(A,n-3),     FILA(A,n-2),     FILA(A,n-1),     FILA(t,(n-2)%6), n);
    err= lap_step_line ( err, FILA(t,(n-4)%6), FILA(t,(n-3)%6), FILA(t,(n-2)%6), FILA(A,n-3),     n);

    err= lap_step_line ( err, FILA(t,(n-3)%6), FILA(t,(n-2)%6), FILA(A,n-1),     FILA(A,n-2),     n);
  }
  err = sqrtf( err );

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, err);
  printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);

  free(A);  free(t);
}

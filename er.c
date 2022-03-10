#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 128

int main() {
    double **u = (double **) calloc(n, sizeof(double *));
    double **up = (double **) calloc(n, sizeof(double *));
    for (int i = 0; i < n; i++) {
        u[i] = (double *) calloc(n, sizeof(double));
        up[i] = (double *) calloc(n, sizeof(double));
    }

    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    u[0][0] = up[0][0] = x1;
    u[0][n-1] = up[0][n-1] = x2;
    u[n-1][0] = up[n-1][0] = y1;
    u[n-1][n-1] = up[n-1][n-1] = y2;

    double step1 = 10.0/(n-1);
//#pragma acc parallel loop independent collapse(1) async(1)
    for (int i = 1; i<n-1; i++){
        u[0][i] = up[0][i] = x1 + i*step1;
        u[n-1][i] = up[n-1][i] = y1 + i*step1;
        u[i][0] = up[i][0] = x1 + i*step1;
        u[i][n-1] = up[i][n-1] = x2 + i*step1;
    }

    int itter = 0;
    double error = 1.0;
#pragma acc data copy(u[0:n][0:n]) create(up[0:n][0:n], error) copyout(error)
    {
        while (itter < 1000000 && error > 1e-6) {
            itter++;
#pragma acc kernels async(1)
            {
                error = 0.0;
#pragma acc loop independent collapse(2) reduction(max:error)
                for (int i = 1; i < n - 1; i++) {
                    for (int j = 1; j < n - 1; j++) {
                        up[i][j] = 0.25 * (u[i][j - 1] + u[i][j + 1] + u[i + 1][j] + u[i - 1][j]);
                        error = fmax(error, fabs(up[i][j] - u[i][j]));
                    }
                }
            }
#pragma acc parallel loop independent collapse(2) async(1)
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    u[i][j] = up[i][j];
                }
            }
            if(itter % 100 == 0 || itter == 1 )
#pragma acc wait(1)
#pragma acc update self(error)
                printf("%d %e\n", itter, error);

        }
    }
    printf("%d\n", itter);
    printf("%e", error);
    return 0;
}

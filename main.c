#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 1024

int main() {
    FILE *fp;
    fp = fopen("f.txt", "wt");
    double u[n][n]={0}, up[n][n]={0};
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            u[i][j] = 0.0;
        }
    }
    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 30.0;
    double y2 = 20.0;
    u[0][0] = up[0][0] = x1;
    u[0][n-1] = up[0][n-1] = x2;
    u[n-1][0] = up[n-1][0] = y1;
    u[n-1][n-1] = up[n-1][n-1] = y2;

    double step1 = (x2 - x1)/(n-1);
    double step1x = step1;
    for (int i = 1; i<n-1; i++){
        u[0][i] = x1 + step1x;
        step1x += step1;
    }

    double step2 = (y2 - y1)/(n-1);
    double step2x = step2;
    for (int i = 1; i<n-1; i++){
        u[n-1][i] = y1 + step2x;
        step2x += step2;
    }

    double step3 = (y1 - x1)/(n-1);
    double step3x = step3;
    for (int i = 1; i<n-1; i++){
        u[i][0] = x1 + step3x;
        step3x += step3;
    }

    for (int i = 1; i<n; i++){
        u[i][n-1] = 20.0;
    }
    int itter = 0;
    double error = 1.0;
#pragma acc data copy(u) create(up)
    while (itter<1000000 && error > 1e-6) {
        {
#pragma acc data present(u, up)
#pragma acc parallel reduction(max:error)
#pragma acc loop independent
            for(int i = 1; i < n-1; i++) {
#pragma acc loop independent
                for (int j = 1; j < n - 1; j++) {
                    up[i][j] = 0.25 * (u[i][j - 1] + u[i][j + 1] + u[i + 1][j] + u[i - 1][j]);
                    error = fmax(error, fabs(u[i][j] - up[i][j]));
                }
            }
        }
#pragma acc parallel
        {
#pragma acc loop independent
            for (int i = 1; i < n - 1; i++)
#pragma acc loop independent
                    for (int j = 1; j < n - 1; j++)
                        u[i][j] = up[i][j];
        }
        itter++;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
           fprintf(fp, "%f ", u[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp,"\n%i", itter);
    fclose(fp);
    return 0;
}
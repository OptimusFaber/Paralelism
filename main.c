#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 256

int main() {
    FILE *f;
    f = fopen("C:\\Users\\titan\\C++\\untitled2\\f.txt", "wt");
    double u[n][n], up[n][n];
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

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%10.3e ", u[i][j]);
        }
        fprintf(f, "\n");
    }

    double a = 1.0;
    double xmax = 1.0;
    double h = xmax/n;
    double t = 30*a/(n*n*n);
    int itter = 0;
    double flag;
    while (itter<1000000) {
        int f = 0;
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                up[i][j] = u[i][j] + a * t * (u[i-1][j] - 2 * u[i][j] + u[i+1][j])/(h*h) + a * t * (u[i][j-1] - 2 * u[i][j] +
                                                                                              u[i][j+1])/(h*h);
                if (fabs(u[i][j] - up[i][j]) > 1e-6) {
                    f = 1;
                    flag = fabs(u[i][j] - up[i][j]);
                }
            }
            for (int j = 1; j < n - 1; j++) {
                u[i][j] = up[i][j];
            }
        }
        itter++;
        printf("%i %e\n", itter, flag);
        if (f == 0){
            break;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%e ", u[i][j]);
        }
        printf("\n");
    }
    printf("%i \n", itter);

    return 0;
    fclose(f);
}
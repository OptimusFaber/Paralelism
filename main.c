#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 128

int main() {
    double u[n], up[n];
    u[0] = 0.0;
    for (int i = 1; i < n; i++) {
        u[i] = 0.0;
    }
    u[n-1] = 1.0;
    up[0] = 0.0;
    up[n-1] = 1.0;
    double a = 1.0;
    double xmax = 1.0;
    double h = xmax/n;
    double t = a/(n*n*n);
    int itter = 0;
    while (1) {
        int f = 0;
        for (int i = 1; i < n - 1; i++) {
            up[i] = u[i] + a * t * (u[i - 1] - 2 * u[i] + u[i + 1]) / (h * h);
//            printf("%e %e \n", u[i], up[i]);
            if (fabs(u[i] - up[i]) > 1e-6) {
                f = 1;
            }
        }
        for (int i = 1; i < n - 1; i++) {
            u[i] = up[i];
        }
        itter++;
        if (f == 0){
            break;
        }
    }

    for (int i = 1; i < n; i++) {
        printf("%e \n", u[i]);
    }
    printf("%i \n", itter);

    return 0;
}

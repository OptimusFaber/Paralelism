#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 128

void one_dim_prog() {
	    double u[n], up[n];
	        u[0] = 0.0;  
		    for (int i = 1; i < n - 1; i++) {
			            u[i] = 0.0;
				        }
		        u[n - 1] = 1.0;
			    up[0] = 0.0;
			        up[n - 1] = 1.0;
				    double x_max = 1.0;
				        double h = x_max / (double)n;
					    double a = 1.0;
					        double tau = a / (n * n * n);


						    int it_num = 0;
						        int flag = 1;
							    while (flag == 1) {
								            flag = 0;
									            for (int i = 1; i < n - 1; i++) {
											                up[i] = u[i] + tau * a * (u[i - 1] - 2 * u[i] + u[i + 1]) / (h * h);
													            if (fabs(u[i] - up[i]) > 1e-6)
															                    flag = 1;
														            }
										            for (int i = 1; i < n - 1; i++) {
												                u[i] = up[i];
														        }
											            it_num++;
												        }
							        for (int i = 0; i < n; i++)
									        printf("%e\n", u[i]);
								    printf("number of iterations: %d", it_num);

}

void two_dim_prog() {
	     
	    double u[n][n] = {0};
	        double u_n[n][n] = {0};
		    u[0][0] = 10;
		        u[0][n - 1] = 20;
			    u[n - 1][0] = 20;
			        u[n - 1][n - 1] = 30;

				    double up = (double)(u[n - 1][0] - u[0][0]) / (n-1);
				        double left = (double)(u[n-1][0] - u[0][0]) / (n-1);
					    double right = (double)(u[n - 1][n - 1] - u[n - 1][0]) /(n-1);
					        double down = (double)(u[n - 1][n - 1] - u[0][n - 1]) / (n-1);

						    for (int i = 1; i < n - 1; i++) {
							            u[0][i] = u[0][i - 1] + up;
								            u[n - 1][i] = u[n - 1][i - 1] + down;
									            u[i][0] = u[i - 1][0] + left;
										            u[i][n - 1] = u[i - 1][n - 1] + right;
											        }

						        int it = 0;
							    double err = 1;
							        #pragma acc data copy(u) create(u_n)
							        while (err > 1e-6 && it < 100000) {

									        err = 0;
										        #pragma acc data present(u,u_n)
										        #pragma acc parallel reduction(max:err)
										        {
#pragma acc loop independent
												            for (int i = 1; i < n - 1; i++) {
#pragma acc loop independent
														                    for (int j = 1; j < n - 1; j++) {
																	                        u_n[i][j] = 0.25 * (u[i][j - 1] + u[i][j + 1] + u[i + 1][j] + u[i - 1][j]);
																				                    err = fmax(err, u_n[i][j] - u[i][j]);
																						                    }
																                }
													            }
#pragma acc parallel
											            {
#pragma acc loop independent
													                    for (int i = 1; i < n - 1; i++)
#pragma acc loop independent
																                        for (int j = 1; j < n - 1; j++)
																				                        u[i][j] = u_n[i][j];
															                }
												            printf("%d\n", it);
													            it++;
														        }

								    for (int i = 0; i < n; i++) {
									            for (int j = 0; j < n; j++)
											                printf("%f ", u[i][j]);
										            printf("\n");
											        }
								        printf("iterations: %d", it);
}

int main() {

	    two_dim_prog();
	        return 0;
}

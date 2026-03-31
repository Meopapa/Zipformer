#ifndef LOSS_H
#define LOSS_H

#include <stdio.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-15

double LOSS_mse(double *y, double *y_pred, int n);
double LOSS_mae(double *y, double *y_pred, int n);
double LOSS_rmse(double *y, double *y_pred, int n);

double LOSS_cross_entropy(double *y, double *y_pred, int n);

#endif
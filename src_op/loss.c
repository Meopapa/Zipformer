#include "D:\AI_C\lib\loss.h" 

double LOSS_mse(double *y, double *y_pred, int n)
{
    assert((y && y_pred) && "Pointer error");
    assert(n && "No size");

    double sum = 0;

    for(int i = 0; i < n; i++) sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);

    return sum / n;
}

double LOSS_mae(double *y, double *y_pred, int n)
{
    assert((y && y_pred) && "Pointer error");
    assert(n && "No size");

    double sum = 0;

    for(int i = 0; i < n ; i++) sum+= fabs(y[i] - y_pred[i]);

    return sum / n;
}

double LOSS_rmse(double *y, double *y_pred, int n)
{
    assert((y && y_pred) && "Pointer error");
    assert(n && "No size");

    double sum = 0;

    for(int i = 0; i < n; i++) sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);

    sum /= n;
    
    sum = sqrt(sum);

    return sum;
}

double LOSS_cross_entropy(double *y, double *y_pred, int n) {
    assert((y && y_pred) && "Pointer error");
    assert(n > 0 && "No size");

    double sum = 0;

    if (n > 1) 
    {
        for (int i = 0; i < n; i++) 
        {
            double pred = y_pred[i] < EPSILON ? EPSILON : (y_pred[i] > 1.0 - EPSILON ? 1.0 - EPSILON : y_pred[i]);
            sum += y[i] * log(pred);
        }
        return -sum; 
    } 
    else 
    {
        for (int i = 0; i < n; i++) 
        {
            double pred = y_pred[i] < EPSILON ? EPSILON : (y_pred[i] > 1.0 - EPSILON ? 1.0 - EPSILON : y_pred[i]);
            sum += y[i] * log(pred) + (1 - y[i]) * log(1 - pred);
        }
        return -sum / n;
    }
}
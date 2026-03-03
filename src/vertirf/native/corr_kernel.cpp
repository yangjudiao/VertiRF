#include <cmath>

extern "C" __declspec(dllexport) void corr_same_double(
    const double* x,
    const double* y,
    int n,
    double* out
) {
    if (n <= 0) {
        return;
    }
    const int center = n / 2;
    for (int i = 0; i < n; ++i) {
        const int lag = i - center;
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            const int k = j - lag;
            if (k >= 0 && k < n) {
                acc += x[j] * y[k];
            }
        }
        out[i] = acc;
    }
}

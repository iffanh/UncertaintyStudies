import numpy as np 
import scipy.stats 
np.random.seed(123)


def raw_moment(x:np.ndarray, A:float, order:int):
    return np.sum((x-A)**order)/x.shape[0]

samples = np.random.uniform(-1, 1, 10)
# samples = np.array([34.0, 35.0, 39.0, 43.0, 46.5, 48.5, 50.0, 51.5, 52.5])

mean = raw_moment(samples, 0, 1)
variance = raw_moment(samples, 0, 2)
skewness = raw_moment(samples, 0, 3)
kurtosis = raw_moment(samples, 0, 4)

print(f"Mean comparison: {mean} and {scipy.stats.moment(samples, 1)}")
print(f"Variance comparison: {variance} and {scipy.stats.moment(samples, 2)}")
print(f"Skewness comparison: {skewness} and {scipy.stats.moment(samples, 3)}")
print(f"Kurtosis comparison: {kurtosis} and {scipy.stats.moment(samples, 4)}")
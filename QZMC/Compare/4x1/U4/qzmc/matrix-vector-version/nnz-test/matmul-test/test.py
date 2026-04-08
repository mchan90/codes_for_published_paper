import numpy as np
from datetime import datetime
import time

# Random 4900 x 4900 matrix and 4900 vector
A = np.random.rand(2450, 2450)
B = np.random.rand(2450, 2450)
v = np.random.rand(2450)

# Measure time for matrix-vector multiplication
#start_time = time.time()
start_time = datetime.now()
for k in range(1,2000):
    v = np.random.rand(2450)
    v = A@v
    v = B@v
#end_time = time.time()
end_time = datetime.now()
elapsed = end_time-start_time
print(str(elapsed.total_seconds())+'seconds')

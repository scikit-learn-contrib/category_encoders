import psutil
import time
import pandas as pd
import numpy as np
import multiprocessing
import category_encoders as encoders
import category_encoders.tests.helpers as th

__author__ = 'LiuShulun'


"""
Benchmarking of cpu utilization during every encoding.
Record the peak value and average value.
Commonly, the value is below 50 because of the core distribution by system.
HashingEncoder

TODO:
differentiate between average and peak
differentiate between training and scoring
"""

# benchmarking result format
result_cols = ['encoder', 'X_shape', 'average_time (s)', 'average_cpu_utilization (%)']
results = []
cpu_utilization = multiprocessing.Manager().Queue()

np_X = th.create_array(n_rows=10000)
np_y = np.random.randn(np_X.shape[0]) > 0.5
X = th.create_dataset(n_rows=10000)
X_t = th.create_dataset(n_rows=5000, extras=True)



def get_cpu_utilization():
    """
    new process for recording cpu utilization
    record cpu utilization every 0.2 second & calculate its mean value
    the value is the cpu utilization during every encoding
    """
    global cpu_utilization
    psutil.cpu_percent(None)
    while True:
        cpu_utilization.put(psutil.cpu_percent(None))
        time.sleep(0.2)


psutil.cpu_percent(None)
for encoder_name in encoders.__all__:
    print(encoder_name)
    rsl = [encoder_name, X.shape]
    enc = getattr(encoders, encoder_name)(cols=['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical'])

    t = []
    c = []
    for index in range(3):
        start = time.time()
        proc = multiprocessing.Process(target=get_cpu_utilization, args=())
        proc.start()
        enc.fit(X, np_y)
        th.verify_numeric(enc.transform(X_t))
        end = time.time()
        proc.terminate()
        proc.join()
        cost = []
        while not cpu_utilization.empty():
            cost.append(cpu_utilization.get())
        t.append(end - start)
        c.append(np.mean(cost))
    rsl.append(np.mean(t))
    rsl.append(np.mean(c))

    results.append(rsl)
    print(rsl)

result_df = pd.DataFrame(results, columns=result_cols)
result_df.to_csv('./output/result.csv')

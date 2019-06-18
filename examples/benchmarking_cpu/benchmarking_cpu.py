import psutil
import time
import pandas as pd
import numpy as np
import multiprocessing
import category_encoders as encoders
import category_encoders.tests.helpers as th

__author__ = 'LiuShulun'


"""
Record the average and peak system-wide CPU utilization during encoder training and scoring.
The utilization is reported as a percentage and as such should always be in the range 0..100%.
E.g.: 50% means half of the logical CPUs are completely utilized in a multi-core device,
or half of the CPU is utilized in a single-core device.

TODO: Add a time counter to terminate training and scoring after timeout.
"""

# sampling rate of cpu utilization, smaller for more accurate
cpu_sampling_rate = 0.2

# loop times of benchmarking in every encoding, larger for more accurate but longer benchmarking time
benchmark_repeat = 3

# sample num of data
data_lines = 10000

# benchmarking result format
result_cols = ['encoder', 'used_processes', 'X_shape', 'min_time(s)', 'average_time(s)', 'max_cpu_utilization(%)', 'average_cpu_utilization(%)']
results = []
cpu_utilization = multiprocessing.Manager().Queue()

# define data_set
np_X = th.create_array(n_rows=data_lines)
np_y = np.random.randn(np_X.shape[0]) > 0.5
X = th.create_dataset(n_rows=data_lines)
X_t = th.create_dataset(n_rows=int(data_lines / 2), extras=True)

cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical']


def get_cpu_utilization():
    """
    new process for recording cpu utilization
    record cpu utilization every [cpu_sampling_rate] second & calculate its mean value
    the value is the cpu utilization during every encoding
    """
    global cpu_utilization
    psutil.cpu_percent(None)
    while True:
        cpu_utilization.put(psutil.cpu_percent(None))
        time.sleep(cpu_sampling_rate)


psutil.cpu_percent(None)
for encoder_name in encoders.__all__:
    """
    HashingEncoder gets more benchmarking for different max_process
    """
    num = multiprocessing.cpu_count() if encoder_name == 'HashingEncoder' else 1

    for index in range(num):
        rsl = [encoder_name, index + 1, X.shape]

        if encoder_name == 'HashingEncoder':
            enc = encoders.HashingEncoder(max_process=index+1, cols=cols)
        else:
            enc = getattr(encoders, encoder_name)(cols=cols)

        t = []
        c = []
        for _ in range(benchmark_repeat):
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
        rsl.append(min(t))
        rsl.append(np.mean(t))
        rsl.append(max(c))
        rsl.append(np.mean(c))

        results.append(rsl)
        print(rsl)

result_df = pd.DataFrame(results, columns=result_cols)
result_df.to_csv('./output/result.csv')

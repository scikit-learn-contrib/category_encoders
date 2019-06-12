import time
import numpy as np
import psutil
import pandas as pd
import multiprocessing
import category_encoders.tests.helpers as th
import category_encoders as encoders


# data definitions
np_X = th.create_array(n_rows=10000)
np_y = np.random.randn(np_X.shape[0]) > 0.5
X = th.create_dataset(n_rows=10000)
X_t = th.create_dataset(n_rows=5000, extras=True)


result_cols = ['encoder', 'X_shape', 'average_time', 'average_cpu_utilization']

results = []

cpu_utilization = multiprocessing.Manager().Queue()


def get_cpu_utilization():
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

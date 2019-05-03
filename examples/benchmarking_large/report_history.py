import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

results_df = pd.read_csv('./output/result_2019-01-05.csv')
previous_df = pd.read_csv('./output/result_2018-09-02.csv')
results_df['version'] = '2019-01'
previous_df['version'] = '2018-09'
appended_df = results_df.append(previous_df)
filtered_df = appended_df[appended_df.encoder != 'GaussEncoder']
filtered_df = filtered_df[filtered_df.encoder != 'MEstimateEncoder']


# Fit runtime by encoder
f = plt.figure(figsize=(15, 9))
sb.barplot(data=filtered_df, x="encoder", y="fit_encoder_time", hue="version", hue_order=['2018-09', '2019-01'])
plt.xlabel('Encoder')
plt.ylabel('Encoder fit runtime [s]')
f.savefig("./output/fit_runtime.pdf", bbox_inches='tight')

# Score runtime by encoder
f = plt.figure(figsize=(15, 9))
sb.barplot(data=filtered_df, x="encoder", y="score_encoder_time", hue="version", hue_order=['2018-09', '2019-01'])
plt.xlabel('Encoder')
plt.ylabel('Encoder score runtime [s]')
f.savefig("./output/score_runtime.pdf", bbox_inches='tight')

# Test AUC by encoder
f = plt.figure(figsize=(15, 9))
sb.barplot(data=filtered_df, x="encoder", y="test_auc", hue="version", hue_order=['2018-09', '2019-01'])
plt.xlabel('Encoder')
plt.ylabel('Testing AUC')
plt.show()
f.savefig("./output/test_auc.pdf", bbox_inches='tight')
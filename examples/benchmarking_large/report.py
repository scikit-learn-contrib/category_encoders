import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

results_df = pd.read_csv('./output/result_2018-09-02.csv')

# Box plot with results grouped by encoder
f = plt.figure(figsize=(15, 9))
sb.boxplot(data=results_df, x='encoder', y='test_auc', notch=True)
plt.grid(True, axis='y')
f.savefig("./output/boxplot_encoder.pdf", bbox_inches='tight')

# The results grouped by encoder + classifier
f = plt.figure(figsize=(12, 12))
for index, clf in enumerate(results_df['model'].unique()):
    plt.subplot(3, 3, index + 1)
    plt.title(clf)
    sb.boxplot(data=results_df.loc[results_df['model'] == clf], y='encoder', x='test_auc', notch=True)
    plt.grid(True, axis='x')
    plt.ylabel('')
    if index < 6 != 0:
        plt.xlabel('')
    if index % 3 != 0:
        plt.yticks([])
    plt.tight_layout()
    plt.xlim(0.0, 1.0)
f.savefig("./output/boxplot_encoder_model.pdf", bbox_inches='tight')
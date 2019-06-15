import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

results_df = pd.read_csv('./output/result.csv')


# AUC grouped by encoder
f = plt.figure(figsize=(9, 9))
sb.boxplot(data=results_df, y='encoder', x='test_auc', notch=True)
plt.grid(True, axis='x')
f.savefig("./output/auc.pdf", bbox_inches='tight')


# AUC grouped by encoder + classifier
f = plt.figure(figsize=(12, 15))
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
f.savefig("./output/auc_model.pdf", bbox_inches='tight')


# Overfitting
df_overfitting = pd.melt(results_df, col_level=0, id_vars=['encoder', 'model'], value_vars=['train_auc', 'test_auc'], value_name='auc')


# Clustered AUC grouped by encoder
f = plt.figure(figsize=(9, 9))
sb.boxplot(data=df_overfitting, y='encoder', x='auc', hue='variable', notch=True)
plt.grid(True, axis='x')
f.savefig("./output/overfitting.pdf", bbox_inches='tight')


# Clustered AUC grouped by encoder + classifier
f = plt.figure(figsize=(12, 15))
for index, clf in enumerate(df_overfitting['model'].unique()):
    plt.subplot(3, 3, index + 1)
    plt.title(clf)
    sb.boxplot(data=df_overfitting.loc[df_overfitting['model'] == clf], y='encoder', x='auc', hue='variable', notch=True)
    plt.grid(True, axis='x')
    plt.ylabel('')
    if index < 6 != 0:
        plt.xlabel('')
    if index % 3 != 0:
        plt.yticks([])
    plt.tight_layout()
    plt.xlim(0.0, 1.0)
f.savefig("./output/overfitting_model.pdf", bbox_inches='tight')

import pandas as pd
import matplotlib.pyplot as plt

filename = "data.txt"
df = pd.read_csv(filename, sep=' ', header=None, names=['N iteration', 'Cost', 'Accuracy'])

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['N iteration'], df['Cost'], color='tab:blue', marker='o', label='Cost')
ax1.set_xlabel("N iteration", fontsize=28)
ax1.set_ylabel("Cost", color='tab:blue', fontsize=28)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(df['N iteration'], df['Accuracy'], color='tab:red', marker='s', label='Accuracy')
ax2.set_ylabel("Accuracy", color='tab:red',  fontsize=28)
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle("Train", fontsize=28)
ax1.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import re

# Список для хранения данных
epochs = []
losses = []
accuracies = []

# Чтение файла и парсинг данных
with open("training_log.txt", "r") as f:
    for line in f:
        # Извлечение значений с помощью регулярных выражений
        epoch_match = re.search(r'Epoch: \[(\d+)/\d+\]', line)
        loss_match = re.search(r'Loss: ([\d.]+)', line)
        acc_match = re.search(r'ValAcc: ([\d.]+%)', line)
        
        if epoch_match and loss_match and acc_match:
            epochs.append(int(epoch_match.group(1)))
            losses.append(float(loss_match.group(1)))
            # Удаление знака % и преобразование в float
            acc_str = acc_match.group(1).replace('%', '')
            accuracies.append(float(acc_str))

# Создание DataFrame с ожидаемыми именами столбцов
df = pd.DataFrame({
    'N iteration': epochs,
    'Cost': losses,
    'Accuracy': accuracies
})

# Построение графиков
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['N iteration'], df['Cost'], color='tab:blue', marker='o', label='Cost')
ax1.set_xlabel("N iteration", fontsize=28)
ax1.set_ylabel("Cost", color='tab:blue', fontsize=28)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(df['N iteration'], df['Accuracy'], color='tab:red', marker='s', label='Accuracy')
ax2.set_ylabel("Accuracy", color='tab:red', fontsize=28)
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle("Train", fontsize=28)
ax1.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()

plt.show()

from matplotlib import pyplot as plt

def plot_loss_per_epoch(train_loss, val_loss):
    epoch = len(train_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='o', label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(epoch+2))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_per_epoch(train_acc, val_acc):
    epoch = len(train_acc)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, marker='o', label='Training Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, marker='o', label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(epoch+2))
    plt.legend()
    plt.grid(True)
    plt.show()
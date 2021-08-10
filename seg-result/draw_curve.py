import numpy as np
import matplotlib.pyplot as plt


def draw(path):
    with open(path, "r") as f:
        lines = f.readlines()
    train_accs, test_accs = [], []
    for line in lines:
        if line.startswith("process"):
            train_accs.append(float(line[-5:]))
        elif line.startswith("test process"):
            test_accs.append(float(line[-5:]))
    train_accs, test_accs = np.array(train_accs), np.array(test_accs)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    x = np.arange(0, 100)
    plt.plot(x, train_accs, marker="o", ms=3, label="train dataset")
    plt.plot(x, test_accs, marker="s", ms=3, label="test dataset")
    plt.legend(loc="lower right")
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.show()


if __name__ == '__main__':
    # draw("./seg-bn.txt")
    with open("./seg.txt", "r") as f:
        lines = f.readlines()
    train_accs, test_accs = [], []
    for line in lines:
        if line.startswith("process"):
            train_accs.append(float(line[-5:]))
        elif line.startswith("test process"):
            test_accs.append(float(line[-5:]))
    train_accs, test_accs = np.array(train_accs), np.array(test_accs)

    with open("./seg-bn.txt", "r") as f:
        lines = f.readlines()
    train_bn_accs, test_bn_accs = [], []
    for line in lines:
        if line.startswith("process"):
            train_bn_accs.append(float(line[-5:]))
        elif line.startswith("test process"):
            test_bn_accs.append(float(line[-5:]))
    train_bn_accs, test_bn_accs = np.array(train_bn_accs), np.array(test_bn_accs)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    x = np.arange(0, 100)
    plt.plot(x, train_accs, marker="o", ms=3, label="train dataset without bn")
    plt.plot(x, test_accs, marker="s", ms=3, label="test dataset without bn")
    plt.plot(x, train_bn_accs, marker="o", ms=3, label="train dataset with bn")
    plt.plot(x, test_bn_accs, marker="s", ms=3, label="test dataset with bn")
    plt.legend(loc="lower right")
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.show()
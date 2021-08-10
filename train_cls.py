import numpy as np
from dataset import Dataloader
from layer import Linear, MaxPool, ReLU
from loss import CrossEntropyLoss

train_batch_size, test_batch_size = 64, 25
learning_rate = 0.01
epoch = 100


class Model:
    def __init__(self, save_path="./params-v1/"):
        self.layers = [
            Linear(3, 64), ReLU(),
            Linear(64, 128), ReLU(),
            MaxPool(), Linear(128, 3)
        ]
        self.save_path = save_path

    def __call__(self, x, start_idx):
        y = x
        for layer in self.layers:
            y = layer(y, start_idx) if isinstance(layer, MaxPool) else layer(y)
        return y

    def grad(self, L_y):
        grad_ = L_y
        for i in range(len(self.layers)-1, -1, -1):
            grad_ = self.layers[i].grad(grad_)

    def optimize(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.w -= lr*layer.w_grad_momutom

    def save(self):
        self.layers[0].save(self.save_path+"linear1.npy")
        self.layers[2].save(self.save_path+"linear2.npy")
        self.layers[5].save(self.save_path+"linear3.npy")


if __name__ == '__main__':
    train_loader, test_loader = Dataloader(batch_size=train_batch_size, mode="train"), Dataloader(batch_size=test_batch_size, mode="test")
    print(len(train_loader.files), len(train_loader))
    print(len(test_loader.files), len(test_loader))
    f = Model()
    loss_fn = CrossEntropyLoss()
    max_acc = 0
    for epoch_count in range(1, epoch+1):
        train_loader.reset()
        correct = 0
        # 训练
        for i in range(len(train_loader)):
            pc, label, start_idx = train_loader.get(i)
            pc = np.concatenate(pc, axis=0)
            y = f(pc, start_idx)
            # print(y.shape)
            loss = loss_fn(y, label)
            y_pred = np.argmax(y, axis=1)
            correct += np.sum(y_pred == label)
            acc = np.sum(y_pred == label) / train_batch_size
            grad = loss_fn.grad()
            f.grad(grad)
            f.optimize(learning_rate)
            print("\rprocess: %d / %d   loss: %.3f  cur acc: %.3f  acc: %.3f" % (i+1, len(train_loader), loss, acc, correct/((i+1)*train_batch_size)), end="")
        # 测试
        print("\ntest...")
        correct = 0
        for i in range(len(test_loader)):
            pc, label, start_idx = test_loader.get(i)
            pc = np.concatenate(pc, axis=0)
            y = f(pc, start_idx)
            # print(y.shape)
            loss = loss_fn(y, label)
            y_pred = np.argmax(y, axis=1)
            correct += np.sum(y_pred == label)
            acc = np.sum(y_pred == label) / test_batch_size
            print("\rtest process: %d / %d   loss: %.3f  cur acc: %.3f  acc: %.3f" % (i+1, len(test_loader), loss, acc, correct/((i+1)*test_batch_size)), end="")
        test_acc = correct / len(test_loader.files)
        print("\nepoch: %d  acc: %.3f" % (epoch_count, test_acc))
        # 保存最优解
        if max_acc < test_acc:
            max_acc = test_acc
            print("max acc: %.3f, save..." % max_acc)
            f.save()
            print("save finish !")

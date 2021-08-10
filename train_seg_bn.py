import numpy as np
from dataset import SegDataloader
from layer import Linear, MaxPool, ReLU, Copy, BatchNorm
from loss import CrossEntropyLoss

train_batch_size, test_batch_size = 64, 25
learning_rate = 0.01
epoch = 100


class Model:
    def __init__(self, save_path="./params-seg-bn/"):
        self.layers = [
            Linear(3, 64), BatchNorm(64), ReLU(),
            Linear(64, 128), BatchNorm(128), ReLU(),
            MaxPool(), Copy()
        ]
        self.seg_layer = Linear(64+128, 11)
        self.save_path = save_path

    def __call__(self, x, start_idx):
        y = x
        # for layer in self.layers[:2]:
        for layer in self.layers[:3]:
            y = layer(y)
        y_64 = y
        # for layer in self.layers[2:]:
        for layer in self.layers[3:]:
            y = layer(y, start_idx) if (isinstance(layer, MaxPool) or isinstance(layer, Copy)) else layer(y)
        y = np.concatenate([y_64, y], axis=1)
        y = self.seg_layer(y)
        return y

    def grad(self, L_y):
        grad_ = L_y
        grad_192 = self.seg_layer.grad(grad_)
        grad_64, grad_ = grad_192[:, :64], grad_192[:, 64:]
        # for i in range(len(self.layers)-1, 1, -1):
        for i in range(len(self.layers) - 1, 2, -1):
            grad_ = self.layers[i].grad(grad_)
        grad_ += grad_64
        # for i in range(1, -1, -1):
        for i in range(2, -1, -1):
            grad_ = self.layers[i].grad(grad_)

    def optimize(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.w -= lr*layer.w_grad_momutom
        self.seg_layer.w -= lr*self.seg_layer.w_grad_momutom

    def save(self):
        self.layers[0].save(self.save_path+"linear1.npy")
        # self.layers[2].save(self.save_path+"linear2.npy")
        self.layers[3].save(self.save_path+"linear2.npy")
        self.layers[1].save(self.save_path+"bn1-mean.npy", self.save_path+"bn1-var.npy")
        self.layers[4].save(self.save_path+"bn2-mean.npy", self.save_path+"bn2-var.npy")
        self.seg_layer.save(self.save_path+"linear3.npy")

    def load(self):
        self.layers[0].load(self.save_path+"linear1.npy")
        self.layers[3].load(self.save_path+"linear2.npy")
        self.layers[1].load(self.save_path+"bn1-mean.npy", self.save_path+"bn1-var.npy")
        self.layers[4].load(self.save_path+"bn2-mean.npy", self.save_path+"bn2-var.npy")
        self.seg_layer.load(self.save_path+"linear3.npy")

    def train(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.mode = "train"

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.mode = "test"


if __name__ == '__main__':
    train_loader, test_loader = SegDataloader(batch_size=train_batch_size, mode="train"), SegDataloader(batch_size=test_batch_size, mode="test")
    print(len(train_loader.files), len(train_loader))
    print(len(test_loader.files), len(test_loader))
    f = Model()
    loss_fn = CrossEntropyLoss()
    max_acc = 0
    for epoch_count in range(1, epoch + 1):
        train_loader.reset()
        correct, point_num = 0, 0
        # 训练
        f.train()
        for i in range(len(train_loader)):
            pc, label, start_idx = train_loader.get(i)
            pc, label = np.concatenate(pc, axis=0), np.concatenate(label, axis=0)
            y = f(pc, start_idx)
            # print(y.shape)
            loss = loss_fn(y, label)
            y_pred = np.argmax(y, axis=1)
            correct += np.sum(y_pred == label)
            point_num += pc.shape[0]
            acc = np.sum(y_pred == label) / pc.shape[0]
            grad = loss_fn.grad()
            f.grad(grad)
            f.optimize(learning_rate)
            print("\rprocess: %d / %d   loss: %.3f  cur acc: %.3f  acc: %.3f" % (i+1, len(train_loader), loss, acc, correct/point_num), end="")
        # 测试
        print("\ntest...")
        correct, point_num = 0, 0
        f.eval()
        for i in range(len(test_loader)):
            pc, label, start_idx = test_loader.get(i)
            pc, label = np.concatenate(pc, axis=0), np.concatenate(label, axis=0)
            y = f(pc, start_idx)
            # print(y.shape)
            loss = loss_fn(y, label)
            y_pred = np.argmax(y, axis=1)
            correct += np.sum(y_pred == label)
            point_num += pc.shape[0]
            acc = np.sum(y_pred == label) / pc.shape[0]
            print("\rtest process: %d / %d   loss: %.3f  cur acc: %.3f  acc: %.3f" % (i+1, len(test_loader), loss, acc, correct/point_num), end="")
        test_acc = correct/point_num
        print("\nepoch: %d  acc: %.3f" % (epoch_count, test_acc))
        # 保存最优解
        if max_acc < test_acc:
            max_acc = test_acc
            print("max acc: %.3f, save..." % max_acc)
            f.save()
            print("save finish !")
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.y_softmax, self.idx = None, None

    def __call__(self, y_pred, y_label):
        # batch_size x cls, batch_size
        batch_size, cls = y_pred.shape
        y_exp = np.exp(y_pred)
        self.y_softmax = y_exp / np.sum(y_exp, axis=1, keepdims=True)
        self.idx = y_label+np.arange(0, batch_size)*cls
        return np.mean(-np.log(self.y_softmax.reshape(-1)[self.idx]))

    def grad(self):
        if self.y_softmax is None:
            print("check your grogram")
        else:
            batch_size, cls = self.y_softmax.shape
            y_pred_grad = self.y_softmax.reshape(-1)
            y_pred_grad[self.idx] = self.y_softmax.reshape(-1)[self.idx] - 1
            self.y_softmax, self.idx = None, None
            return (y_pred_grad.reshape(batch_size, cls)) / batch_size


if __name__ == '__main__':
    loss_fn = CrossEntropyLoss()
    y_pred = np.random.rand(3, 5)
    y_label = np.array([4, 2, 3])
    loss = loss_fn(y_pred, y_label)
    grad = loss_fn.grad()
    print(loss, grad)
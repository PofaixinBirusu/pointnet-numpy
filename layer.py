import numpy as np


class MaxPool:
    def __init__(self):
        self.max_idx = None
        self.start_idx = None

    def __call__(self, x, start_idx):
        result = []
        self.max_idx = []
        self.start_idx = start_idx
        for i in range(len(start_idx)-1):
            pc = x[start_idx[i]:start_idx[i+1]]
            point_num, channel = pc.shape
            max_idx = np.argmax(pc, axis=0)
            max_val = pc.T.reshape(-1)[max_idx+np.arange(0, channel)*point_num]
            self.max_idx.append(max_idx)
            result.append(max_val.reshape(1, -1))
        return np.concatenate(result, axis=0)

    def grad(self, x):
        if self.max_idx is None:
            print("check your program")
        else:
            channel, grads_ = x.shape[1], []
            for i in range(len(self.max_idx)):
                max_idx, point_num = self.max_idx[i], self.start_idx[i+1]-self.start_idx[i]
                grad_ = np.zeros(shape=(channel * point_num))
                grad_[max_idx+np.arange(0, channel)*point_num] = x[i]
                grads_.append(grad_.reshape(channel, point_num).T)
            grads_ = np.concatenate(grads_, axis=0)
            self.max_idx, self.start_idx = None, None
            return grads_


class Linear:
    def __init__(self, in_channel, out_channel):
        self.w = np.random.randn(in_channel, out_channel)*np.sqrt(1/out_channel)
        self.x = None
        self.w_grad = None
        self.w_grad_momutom = None

    def __call__(self, x):
        self.x = x
        return x.dot(self.w)

    def grad(self, L_y):
        if self.x is None:
            print("check your program")
        else:
            self.w_grad = self.x.T.dot(L_y)
            if self.w_grad_momutom is None:
                self.w_grad_momutom = self.w_grad
            else:
                self.w_grad_momutom = 0.9*self.w_grad_momutom+0.1*self.w_grad
            self.x = None
            return L_y.dot(self.w.T)

    def save(self, path):
        np.save(path, self.w)

    def load(self, path):
        self.w = np.load(path)


class ReLU:
    def __init__(self):
        self.g = None

    def __call__(self, x):
        y = np.copy(x)
        y[y < 0] = 0
        self.g = np.ones(shape=x.shape)
        self.g[y < 0] = 0
        return y

    def grad(self, L_y):
        if self.g is None:
            print("check your program")
        else:
            return self.g*L_y


class Copy:
    def __init__(self):
        self.start_idx = None

    def __call__(self, x, start_idx):
        result = []
        self.start_idx = start_idx
        for i in range(len(start_idx)-1):
            n = start_idx[i+1]-start_idx[i]
            result.append(np.tile(x[i].reshape(1, -1), (n, 1)))
        return np.concatenate(result, axis=0)

    def grad(self, L_y):
        if self.start_idx is None:
            print("check your program")
        else:
            grad_ = []
            for i in range(len(self.start_idx)-1):
                grad_.append(np.sum(L_y[self.start_idx[i]:self.start_idx[i+1]], axis=0))
            return np.stack(grad_, axis=0)


class BatchNorm:
    # data: batch_size x c
    def __init__(self, in_channel):
        self.eps = 1e-5
        self.gamma = np.ones(shape=(in_channel, ))
        self.beta = np.zeros(shape=(in_channel, ))

        self.xi = None
        self.mean = None
        self.var = None

        self.data_mean, self.data_var = None, None
        self.mode = "train"

    def __call__(self, xi):
        # forward
        mean = np.mean(xi, axis=0, keepdims=True)
        var = np.var(xi, axis=0, keepdims=True)

        if self.mode == "train":
            xi_hat = (xi - mean) / np.sqrt(var + self.eps)
            # 指数加权
            self.data_mean = mean if self.data_mean is None else 0.9 * self.data_mean + 0.1 * mean
            self.data_var = var if self.data_var is None else 0.9 * self.data_var + 0.1 * var
        else:
            xi_hat = (xi - self.data_mean) / np.sqrt(self.data_var + self.eps)
        yi = self.gamma.reshape(1, -1) * xi_hat + self.beta.reshape(1, -1)
        self.xi, self.mean, self.var, self.yi = xi, mean, var, yi
        return yi

    def grad(self, L_y):
        xi_hat_ = L_y*self.gamma.reshape(1, -1)
        # c
        var_ = np.sum(xi_hat_*(self.xi-self.mean.reshape(1, -1))*(-1/2)*np.power(self.var+self.eps, -3/2), axis=0)
        # c
        mean_ = np.sum(xi_hat_*(-1/np.sqrt(self.var+self.eps)), axis=0)+var_*(np.mean(-2*(self.xi-self.mean), axis=0))
        xi_ = xi_hat_/np.sqrt(self.var+self.eps)+2*var_.reshape(1, -1)*((self.xi-self.mean)/self.xi.shape[0])+mean_.reshape(1, -1)/self.xi.shape[0]
        return xi_

    def save(self, path_mean, path_var):
        np.save(path_mean, self.data_mean)
        np.save(path_var, self.data_var)

    def load(self, path_mean, path_var):
        self.data_mean, self.data_var = np.load(path_mean), np.load(path_var)


if __name__ == '__main__':
    x = np.array([
        [4, 5, 7, 2],
        [1, 7, 3, 0],
        [8, 3, 5, 1]
    ])
    start_idx = [0, 2, 3]
    maxpool = MaxPool()
    result = maxpool(x, start_idx)
    print(result)
    maxpool.max_idx = [[0, 2, 1]]
    maxpool.start_idx = [0, 5]
    x = np.array([[4, 5, 7]])
    print(maxpool.grad(x))
import os
import numpy as np
import json
from random import shuffle


class Dataloader:
    def __init__(self, batch_size, root="E:/shapenet_benchmark_v0_normal", cls_=("03624134", "03948459", "03790512"), mode="train"):
        self.files, all_files = [], []
        for i in range(len(cls_)):
            for _, _, ls in os.walk(root+"/"+cls_[i]):
                all_files += [{"path": cls_[i]+"/"+path, "cls": i} for path in ls]
                break
        self.batch_size = batch_size
        with open(root+"/train_test_split/shuffled_test_file_list.json", "r") as f:
            test_files = set([test_file[11:]+".txt" for test_file in json.load(f)])
        if mode == "train":
            self.files = [{"path": root+"/"+file["path"], "cls": file["cls"]} for file in all_files if file["path"] not in test_files]
        elif mode == "test":
            self.files = [{"path": root+"/"+file["path"], "cls": file["cls"]} for file in all_files if file["path"] in test_files]
        # print(self.files)

    def __len__(self):
        return len(self.files) // self.batch_size

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def reset(self):
        shuffle(self.files)

    def get(self, i):
        strat_idx = i*self.batch_size
        pcs, cls_, cnt = [], [], []
        for j in range(strat_idx, strat_idx+self.batch_size):
            cls_.append(self.files[j]["cls"])
            with open(self.files[j]["path"], "r") as f:
                pc = np.array([[float(x) for x in l.split(" ")[:3]] for l in f.readlines()])
                pcs.append(self.pc_normalize(pc))
                cnt.append(pc.shape[0])
        strat_idx = [0]
        for i in range(len(pcs)):
            strat_idx.append(strat_idx[i]+pcs[i].shape[0])
        return pcs, np.array(cls_), strat_idx


class SegDataloader:
    def __init__(self, batch_size, root="E:/shapenet_benchmark_v0_normal", cls_=("03624134", "03948459", "03790512"), mode="train"):
        self.files, all_files = [], []
        for i in range(len(cls_)):
            for _, _, ls in os.walk(root+"/"+cls_[i]):
                all_files += [{"path": cls_[i]+"/"+path, "cls": i} for path in ls]
                break
        self.batch_size = batch_size
        with open(root+"/train_test_split/shuffled_test_file_list.json", "r") as f:
            test_files = set([test_file[11:]+".txt" for test_file in json.load(f)])
        if mode == "train":
            self.files = [{"path": root+"/"+file["path"], "cls": file["cls"]} for file in all_files if file["path"] not in test_files]
        elif mode == "test":
            self.files = [{"path": root+"/"+file["path"], "cls": file["cls"]} for file in all_files if file["path"] in test_files]
        # print(self.files)
        self.seg_label = {x: i for i, x in enumerate([22, 23, 30, 31, 32, 33, 34, 35, 38, 39, 40])}

    def __len__(self):
        return len(self.files) // self.batch_size

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def reset(self):
        shuffle(self.files)

    def get(self, i):
        strat_idx = i*self.batch_size
        pcs, cls_, cnt = [], [], []
        for j in range(strat_idx, strat_idx+self.batch_size):
            # cls_.append(self.files[j]["cls"])
            with open(self.files[j]["path"], "r") as f:
                pc = np.array([[float(x) for x in l.split(" ")] for l in f.readlines()])
                pcs.append(self.pc_normalize(pc[:, :3]))
                cnt.append(pc.shape[0])
            cls_.append([self.seg_label[int(x)] for x in list(pc[:, -1])])
        strat_idx = [0]
        for i in range(len(pcs)):
            strat_idx.append(strat_idx[i]+pcs[i].shape[0])
        return pcs, cls_, strat_idx


if __name__ == '__main__':
    train_loader = Dataloader(batch_size=8, mode="train")
    test_loader = Dataloader(batch_size=8, mode="test")
    print(len(train_loader.files), len(train_loader))
    print(len(test_loader.files), len(test_loader))
    # for i in range(len(dataloader)):
    #     pc, label, strat_idx = dataloader.get(i)
    #     pc = np.concatenate(pc, axis=0)
    #     print(pc.shape)
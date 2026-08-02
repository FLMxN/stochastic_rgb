from random import uniform
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn   
import math
import numpy as np
import torch

x = []
y = []
rgba= []

rng = np.random.default_rng()

def sigmoid(x):
    return 1/(1+(math.e**-x))
def normalize(x):
    return [2*(p - min(x)) / (max(x) - min(x)) - 1 for p in x]
def scale(x, m):
    return [p/float(m) for p in x]

class Floor:
    def __init__(self) -> None:
        pass
    def climb(self, ranges):
        return torch.from_numpy((np.array([rng.uniform(0+ranges[0], 1-ranges[1]), rng.uniform(0+ranges[2], 1-ranges[3]), rng.uniform(0+ranges[4], 1-ranges[5])])))


class Engine:
    def __init__(self, t, lr) -> None:
        self.lr = lr
        self.layers = [Floor() for _ in range(5)]
        self.t = torch.from_numpy(np.asarray(scale(t, 255)))
        self.w = torch.tensor(0, dtype=torch.float32)
        self.best = torch.tensor(0, dtype=torch.float32)

    def simulation(self):
        leftr = rightr = leftg = rightg = leftb = rightb = 0
        layer_num = 0
        layer_patience = 0
        step = 0
        mse = nn.MSELoss(reduction='none')
        while layer_num < len(self.layers):
            step += 1

            model_loss = min(model_loss, mse(self.w, self.t).mean()) if step>1 else torch.tensor(1, dtype=torch.float64)
            cos_sim = max(cos_sim, F.cosine_similarity(self.t, self.w, dim=0).mean()) if step>1 else 0

            self.w = self.layers[layer_num].climb(ranges=[leftr, rightr, leftg, rightg, leftb, rightb])
            if model_loss > mse(self.w, self.t).mean() and cos_sim < F.cosine_similarity(self.t, self.w, dim=0).mean():
                layer_num += 1
                self.best = self.w
                leftr = rightr = leftg = rightg = leftb = rightb = layer_patience = 0
            else:
                if mse(self.w, self.t).numpy()[0] == max(mse(self.w, self.t).numpy()):
                    leftr += self.lr*(self.best[0])
                    rightr += self.lr*(1-self.best[0])
                elif mse(self.w, self.t).numpy()[1] == max(mse(self.w, self.t).numpy()):
                    leftg += self.lr*(self.best[1])
                    rightg += self.lr*(1-self.best[1])
                elif mse(self.w, self.t).numpy()[2] == max(mse(self.w, self.t).numpy()):
                    leftb += self.lr*(self.best[2])
                    rightb += self.lr*(1-self.best[2])
                layer_patience += 1
                if layer_patience == 1/self.lr: leftr = rightr = leftg = rightg = leftb = rightb = layer_patience = 0

            model_loss = min(model_loss, mse(self.w, self.t).mean())
            print('step=', step,layer_num, ' patience=',layer_patience, " best=", round(float(model_loss.numpy()), 5), ' -- loss=',round(float(mse(self.w, self.t).mean().numpy()), 5), 'tensor=', [round(p, 3) for p in self.w.numpy().tolist()])
            x.append(step)
            y.append(F.cosine_similarity(self.w.float(), self.t.float(), dim=0).mean().item())
            rgba.append(self.w.numpy().tolist() + [step])
            if mse(self.w, self.t).mean() <= self.lr: break 
            # print(self.w.numpy().tolist())


net = Engine(t=[116, 186, 102], lr=0.0001)
net.simulation()

plt.figure(figsize=(8, 8))
sizes = [25] * (len(x) - 1) + [125]
alpha = scale([p[3] for p in rgba], len(x))
rgba = [[rgba[p][0], rgba[p][1],rgba[p][2], alpha[p]] for p in range(len(alpha))]
plt.scatter(x, y, c=rgba, s=sizes)

plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.grid(False)
plt.show()

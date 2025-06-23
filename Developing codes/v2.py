import numpy as np
import pandas as pd
from catsim import irt

class ALManual:
    def __init__(self, name: str, item_bank_file="item_bank.csv", k=10):
        self.name = name
        self.k = k
        self.theta = -3.0
        self.administered_items = []
        self.responses = []
        self.item_bank = irt.normalize_item_bank(pd.read_csv(item_bank_file).values)

    def select_next_item(self, n=1):
        info_values = irt.inf_hpc(self.theta, self.item_bank)
        info_values[self.administered_items] = -np.inf  # mask administered
        top_indices = np.argsort(info_values)[-n:]
        return top_indices.tolist()

    def administer_items(self, items, responses):
        self.administered_items.extend(items)
        self.responses.extend(responses)
        self.theta = self.estimate_theta()

    def estimate_theta(self):
        used_items = self.item_bank[self.administered_items]
        used_responses = self.responses

        def neg_log_likelihood(theta):
            a, b, c = used_items[:, 0], used_items[:, 1], used_items[:, 2]
            P = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
            P = np.clip(P, 1e-6, 1 - 1e-6)
            logL = used_responses * np.log(P) + (1 - np.array(used_responses)) * np.log(1 - P)
            return -np.sum(logL)

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(neg_log_likelihood, bounds=(-6, 6), method='bounded')
        return res.x

    def should_stop(self):
        return len(self.administered_items) >= self.k

    def info(self):
        return {
            "name": self.name,
            "theta": self.theta,
            "administered_items": self.administered_items,
            "responses": self.responses,
        }


al = ALManual("Arpit")


items = al.select_next_item(n=3)
responses = [1.0, 1.0, 1.0]  # simulate correct response
al.administer_items(items, responses)
print(al.info())

items = al.select_next_item(n=2)
responses = [0.0, 0.0]  # simulate mixed responses
al.administer_items(items, responses)
print(al.info())




items = al.select_next_item(n=3)
responses = [1.0, 1.0, 1.0]  # simulate correct response
al.administer_items(items, responses)
print(al.info())

items = al.select_next_item(n=2)
responses = [0.0, 0.0]  # simulate mixed responses
al.administer_items(items, responses)
print(al.info())


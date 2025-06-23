import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt

class ALManualMIRT:
    def __init__(self, name, item_bank_file="item_bank.csv", k=10, d=3):
        self.name = name
        self.k = k
        self.d = d
        self.theta = np.full(d, -3.0)  # Ability vector
        self.administered_items = set()
        self.responses = []
        self.theta_history = []
        self.item_bank = pd.read_csv(item_bank_file).values
        if self.item_bank.shape[1] != self.d + 2:
            raise ValueError(f"Item bank must have {self.d + 2} columns (a1..ad, b, c)")
        self.num_items = self.item_bank.shape[0]

    def select_next_item(self, n=1):
        mask = np.ones(self.num_items, dtype=bool)
        mask[list(self.administered_items)] = False
        info_values = np.full(self.num_items, -np.inf)
        for idx, item in enumerate(self.item_bank):
            if not mask[idx]:
                continue
            a_vec = item[:self.d]
            b, c = item[self.d], item[self.d+1]
            z = np.dot(a_vec, self.theta) - b
            P = c + (1 - c) * expit(z)
            info = np.sum((a_vec**2) * (P * (1 - P)) / ((1 - c)**2))
            info_values[idx] = info
        available_indices = np.where(mask)[0]
        if len(available_indices) == 0:
            return []
        n = min(n, len(available_indices))
        top_indices = available_indices[np.argsort(info_values[available_indices])[-n:]]
        return top_indices.tolist()

    def administer_items(self, items, responses):
        if len(items) != len(responses):
            raise ValueError("Length of items and responses must match.")
        if any(i in self.administered_items for i in items):
            raise ValueError("One or more items have already been administered.")
        self.administered_items.update(items)
        self.responses.extend(responses)
        self.theta = self.estimate_theta()
        self.theta_history.append(self.theta.copy())

    def estimate_theta(self):
        if not self.administered_items:
            return self.theta
        used_indices = list(self.administered_items)
        used_items = self.item_bank[used_indices]
        used_responses = [self.responses[i] for i in range(len(used_indices))]
        def neg_log_likelihood(theta_vec):
            logL = 0
            for item, resp in zip(used_items, used_responses):
                a_vec = item[:self.d]
                b, c = item[self.d], item[self.d+1]
                z = np.dot(a_vec, theta_vec) - b
                P = c + (1 - c) * expit(z)
                P = np.clip(P, 1e-6, 1 - 1e-6)
                logL += resp * np.log(P) + (1 - resp) * np.log(1 - P)
            return -logL
        res = minimize(neg_log_likelihood, self.theta, bounds=[(-6, 6)] * self.d)
        return res.x

    def should_stop(self):
        return self.theta.max() >= 3.0 

    def info(self):
        return {
            "name": self.name,
            "theta": self.theta.copy(),
            "administered_items": sorted(self.administered_items),
            "responses": self.responses.copy(),
        }
    
    def plot_theta_history(self, true_abilities=None):
        theta_array = np.array(self.theta_history)
        steps = np.arange(1, theta_array.shape[0] + 1)
        plt.figure(figsize=(10, 6))
        for dim in range(theta_array.shape[1]):
            plt.plot(steps, theta_array[:, dim], marker='o', label=f'Estimated Theta {dim+1}')
            if true_abilities is not None:
                plt.plot(steps, true_abilities[:, dim], linestyle='--', label=f'True Theta {dim+1}')
        plt.title('Theta Progression Over Time')
        plt.xlabel('Test Step')
        plt.ylabel('Theta Estimate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


al = ALManualMIRT("Arpit", item_bank_file="item_bank.csv", k=6, d=3)

# Store true ability for plotting
true_ability = []
for i in range(100):
    scaled_i = min(i, 99)
    p_correct = 0.40 + (0.50 * scaled_i / 99)  # Starts at 0.40, maxes out at 0.90
    true_theta = -3 + 6 * (scaled_i / 99)
    true_ability.append([true_theta] * 3)
    
    items = al.select_next_item(n=10)
    if not items:
        print("No more items to administer.")
        break
    
    responses = np.random.binomial(1, p_correct, size=len(items)).tolist()
    al.administer_items(items, responses)

al.plot_theta_history(true_abilities=np.array(true_ability[:len(al.theta_history)]))



# 3PL IS NOT PERFORMING WELL ON THIS BECAUSE WE HAVE ONLY DIFFICULTY AND OTHERS 
# ARE JUST ASSUMPTIONS WHICH MAKE CONFUSIONS EVERYWHERE LETS TRY 1PL
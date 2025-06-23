import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt


class ALManualIRT:
    def __init__(self, name, item_bank_file="item_bank.csv"):
        self.name = name
        self.theta = 0.0 
        self.administered_items = set()
        self.responses = []
        self.theta_history = []

        self.item_bank = pd.read_csv(item_bank_file).values
        if self.item_bank.shape[1] < 2:
            raise ValueError("Item bank must have at least 2 columns: 'a' and 'b'")
        self.num_items = self.item_bank.shape[0]

    def select_next_item(self, n=1):
        mask = np.ones(self.num_items, dtype=bool)
        mask[list(self.administered_items)] = False

        info_values = np.full(self.num_items, -np.inf)
        for idx, item in enumerate(self.item_bank):
            if not mask[idx]:
                continue
            a, b = item[0], item[1]
            z = a * self.theta - b
            P = expit(z)
            info = a ** 2 * P * (1 - P)
            info_values[idx] = info

        available_indices = np.where(mask)[0]
        if len(available_indices) == 0:
            return []

        n = min(n, len(available_indices))
        top_indices = available_indices[np.argsort(info_values[available_indices])[-n:]]
        return top_indices.tolist()

    def administer_items(self, items, responses):
        if len(items) != len(responses):
            raise ValueError("Items and responses must match in length.")
        if any(i in self.administered_items for i in items):
            raise ValueError("Some items already administered.")

        self.administered_items.update(items)
        self.responses.extend(responses)
        self.theta = self.estimate_theta()
        self.theta_history.append(self.theta)

    def estimate_theta(self):
        if not self.administered_items:
            return self.theta

        used_indices = list(self.administered_items)
        used_items = self.item_bank[used_indices]
        used_responses = self.responses

        def neg_log_likelihood(theta):
            logL = 0
            for item, resp in zip(used_items, used_responses):
                a, b = item[0], item[1]
                z = a * theta - b
                P = expit(z)
                P = np.clip(P, 1e-6, 1 - 1e-6)
                logL += resp * np.log(P) + (1 - resp) * np.log(1 - P)
            return -logL

        res = minimize(neg_log_likelihood, self.theta, bounds=[(-6, 6)])
        return res.x[0]

    def plot_theta_history(self):
        steps = np.arange(1, len(self.theta_history) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, self.theta_history, marker='o', label="Estimated Theta")
        plt.title("Theta Progression")
        plt.xlabel("Test Step")
        plt.ylabel("Theta")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# # --- Simulation: Alternate Test Pass and Fail --- #
# al = ALManualIRT("Arpit", item_bank_file="item_bank.csv")

# for step in range(10):  # 10 test steps
#     items = al.select_next_item(n=10)
#     if not items:
#         print("No more items to administer.")
#         break

#     # Alternate between all 1s and all 0s each step
#     response_value = 1 if step % 2 == 0 else 0
#     responses = [response_value] * len(items)

#     al.administer_items(items, responses)

# al.plot_theta_history()



# --- Simulation: First 8 Correct, Last 2 Incorrect --- #
al = ALManualIRT("Arpit", item_bank_file="item_bank.csv")

for step in range(10):  # 10 test steps
    items = al.select_next_item(n=10)
    if not items:
        print("No more items to administer.")
        break

    # First 8 correct, last 2 incorrect
    responses = [1]*8 + [0]*2
    responses = responses[:len(items)]  # In case fewer than 10 items left

    al.administer_items(items, responses)

al.plot_theta_history()


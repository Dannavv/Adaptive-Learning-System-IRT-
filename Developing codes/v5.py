import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm

class ALManualIRT:
    def __init__(self, name, item_bank_file="item_bank.csv", discrimination=1.0):
        self.name = name
        self.theta = None
        self.theta_se = None
        self.discrimination = discrimination
        self.administered_items = set()
        self.responses = []
        self.theta_history = []
        self.se_history = []

        self._load_item_bank(item_bank_file)
        self._cache_item_difficulties()
        self.reset_test()  # Record initial theta

    def _load_item_bank(self, item_bank_file):
        try:
            item_bank_df = pd.read_csv(item_bank_file)
            if item_bank_df.shape[1] == 1:
                self.item_difficulties = item_bank_df.iloc[:, 0].values
            else:
                self.item_difficulties = item_bank_df.iloc[:, 1].values
                if not np.allclose(item_bank_df.iloc[:, 0], self.discrimination, atol=0.1):
                    warnings.warn("Ignoring discrimination parameters; using fixed a=1.0")
            if np.any(np.isnan(self.item_difficulties)):
                raise ValueError("Item bank contains NaN values")
            self.num_items = len(self.item_difficulties)
            if self.num_items == 0:
                raise ValueError("Item bank is empty")
        except Exception as e:
            raise ValueError(f"Error loading item bank: {e}")

    def _cache_item_difficulties(self):
        self.difficulties = self.item_difficulties.copy()

    def _calculate_probability(self, theta, difficulty):
        z = self.discrimination * theta - difficulty
        return expit(np.clip(z, -50, 50))

    def _calculate_information(self, theta, difficulty):
        P = self._calculate_probability(theta, difficulty)
        return (self.discrimination ** 2) * P * (1 - P)

    def select_next_item(self, n=1):
        if n <= 0:
            return []
        available_mask = np.ones(self.num_items, dtype=bool)
        if self.administered_items:
            available_mask[list(self.administered_items)] = False
        available_indices = np.where(available_mask)[0]
        if len(available_indices) == 0:
            return []
        info_values = np.array([
            self._calculate_information(self.theta, self.difficulties[idx])
            for idx in available_indices
        ])
        if np.all(info_values == 0):
            return np.random.choice(available_indices, n, replace=False).tolist()
        top_item_positions = np.argsort(info_values)[-n:]
        selected_indices = available_indices[top_item_positions]
        return selected_indices.tolist()

    def administer_items(self, items, responses):
        if len(items) != len(responses):
            raise ValueError("Mismatch between items and responses")
        if not items:
            return
        duplicate_items = [item for item in items if item in self.administered_items]
        if duplicate_items:
            raise ValueError(f"Items already administered: {duplicate_items}")
        invalid_items = [item for item in items if item < 0 or item >= self.num_items]
        if invalid_items:
            raise ValueError(f"Invalid item indices: {invalid_items}")
        if not all(resp in [0, 1] for resp in responses):
            raise ValueError("Responses must be 0 or 1")

        self.administered_items.update(items)
        self.responses.extend(responses)
        self.theta, self.theta_se = self._estimate_theta_with_se()
        self.theta_history.append(self.theta)
        self.se_history.append(self.theta_se)

    def _estimate_theta_with_se(self):
        if not self.administered_items:
            return self.theta, None
        admin_indices = list(self.administered_items)
        admin_difficulties = self.difficulties[admin_indices]
        admin_responses = np.array(self.responses)

        def neg_log_likelihood(theta):
            logL = 0
            for b, r in zip(admin_difficulties, admin_responses):
                P = np.clip(self._calculate_probability(theta, b), 1e-10, 1 - 1e-10)
                logL += r * np.log(P) + (1 - r) * np.log(1 - P)
            return -logL

        def grad(theta):
            return -sum(self.discrimination * (r - self._calculate_probability(theta, b))
                        for b, r in zip(admin_difficulties, admin_responses))

        try:
            result = minimize(
                neg_log_likelihood,
                x0=self.theta,
                method='L-BFGS-B',
                jac=grad,
                bounds=[(-6, 6)],
                options={'ftol': 1e-9, 'gtol': 1e-9}
            )
            theta_est = float(result.x[0])
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            theta_est = self.theta

        try:
            total_info = sum(self._calculate_information(theta_est, b) for b in admin_difficulties)
            se = 1.0 / np.sqrt(total_info) if total_info > 0 else None
        except:
            se = None
        return theta_est, se

    def reset_test(self):
        self.theta = -3.0
        self.theta_se = 0.23
        self.administered_items = set()
        self.responses = []
        self.theta_history = [self.theta]  # Include initial theta
        self.se_history = [self.theta_se]           # Include initial SE

    def get_test_info(self):
        return {
            'name': self.name,
            'items_administered': len(self.administered_items),
            'total_items': self.num_items,
            'current_theta': self.theta,
            'theta_se': self.theta_se,
            'responses': self.responses.copy(),
            'proportion_correct': np.mean(self.responses) if self.responses else None
        }

    def plot_theta_history(self, show_se=True):
        if not self.theta_history:
            print("No theta history to plot.")
            return

        steps = np.arange(len(self.theta_history))
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.theta_history, marker='o', linewidth=2, label="Estimated Theta")

        if show_se and any(se is not None for se in self.se_history):
            theta_array = np.array(self.theta_history)
            se_array = np.array([se if se is not None else 0 for se in self.se_history])
            valid = se_array > 0
            confidence = 0.95
            z = norm.ppf(1 - (1 - confidence) / 2)
            plt.fill_between(
                steps[valid],
                (theta_array - z * se_array)[valid],
                (theta_array + z * se_array)[valid],
                alpha=0.2, color='gray', label=f"{int(confidence*100)}% Confidence Interval"
            )
    

        plt.title(f"Theta Progression for {self.name}")
        plt.xlabel("Test Step")
        plt.ylabel("Theta (Ability Estimate)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        if self.theta_se is not None:
            print(f"Final Theta: {self.theta:.4f} ± {self.theta_se:.4f}")
        else:
            print(f"Final Theta: {self.theta:.4f}")


# --- Run Simulation ---
al = ALManualIRT("Arpit", item_bank_file="item_bank.csv")

print("\nRunning simulation...")
for step in range(10):
    items = al.select_next_item(n=10)
    if not items:
        print("No more items to administer.")
        break
    
    num_ones = int(0.8 * len(items))
    responses = [1] * num_ones + [0] * (len(items) - num_ones)
    np.random.shuffle(responses)


    al.administer_items(items, responses)
    if al.theta_se is not None:
        print(f"Step {step+1}: Theta = {al.theta:.4f}, SE = {al.theta_se:.4f}")
    else:
        print(f"Step {step+1}: Theta = {al.theta:.4f}, SE = N/A")

# Print test info and plot
test_info = al.get_test_info()
print(f"\nFinal Test Statistics:")
print(f"Items administered: {test_info['items_administered']}")
print(f"Proportion correct: {test_info['proportion_correct']:.3f}")

al.plot_theta_history(show_se=True)

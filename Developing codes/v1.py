import os
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_DISABLE_VML'] = '1'

import numexpr
numexpr.set_vml_num_threads(1)
numexpr.set_num_threads(1)


import pandas as pd
import numpy as np
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MaxItemStopper


class AL:
    def __init__(self, name: str, item_bank_file="item_bank.csv", k=10):
        self.name = name
        self.initializer = FixedPointInitializer(-3)
        self.selector = MaxInfoSelector()
        self.estimator = NumericalSearchEstimator()
        self.stopper = MaxItemStopper(k)
        self.item_bank = pd.read_csv(item_bank_file).values

        # Ensure native Python float
        self.theta = float(self.initializer.initialize(1))
        self.administered_items = []
        self.responses = []

    def select_next_item(self, n=1):
        items = []
        temp_administered = self.administered_items.copy()
        for _ in range(n):
            item = self.selector.select(0, self.item_bank, temp_administered, self.theta)
            item = int(item)  # Ensure native int
            items.append(item)
            temp_administered.append(item)
        return items

    def administer_items(self, items, responses):
        # Ensure both lists are native Python types
        items = [int(i) for i in items]
        responses = [float(r) for r in responses]

        self.administered_items.extend(items)
        self.responses.extend(responses)

        # Convert all inputs to native types
        try:
            self.theta = float(self.estimator.estimate(
                0,
                self.item_bank,
                list(map(int, self.administered_items)),
                list(map(float, self.responses)),
                float(self.theta)
            ))
        except Exception as e:
            print("Estimation failed:", e)

    def info(self):
        return {
            "name": self.name,
            "theta": self.theta,
            "administered_items": self.administered_items,
            "responses": self.responses,
        }


# Example usage
al = AL("Arpit")

next_items = al.select_next_item(3)
print(f"Next item to administer: {next_items}")

responses = [1.0] * len(next_items)  # Simulated correct responses
al.administer_items(next_items, responses)
print(al.info())

next_items = al.select_next_item(2)
responses = [0.0] * len(next_items)  # Simulated incorrect responses
al.administer_items(next_items, responses)
print(al.info())


# Estimation failed: couldn't find matching opcode for 'where_dddd'
# I DONT' KNOW WHAT IS THIS PROBLEM 
# IT'S SOME INTERNAL NUMPY PROBLEM SO 
# NOW I NEED TO WRITE CUSTOM FUCNTIONS USING LOW LEVEL FUNCTION OF CATSIM.IRT
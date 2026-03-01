import numpy as np
from skopt import Optimizer

class BayesianOrchestrator:
    def __init__(self, hardware_list):
        
        self.hardware_list = hardware_list
        self.num_hw = len(hardware_list)
    
        self.opt = Optimizer(
            dimensions=[(0, self.num_hw - 1)],
            base_estimator="GP", 
            n_initial_points=3
        )

    def decide(self, task_features):
        
        # Prediciton
        suggested_hw_index = self.opt.ask()[0]
        return suggested_hw_index, self.hardware_list[suggested_hw_index]

    def report_performance(self, hw_index, execution_time, energy_used, weight_efficiency):
        
        # Feedback loop
        cost = execution_time + (energy_used * weight_efficiency)
        self.opt.tell([hw_index], cost)
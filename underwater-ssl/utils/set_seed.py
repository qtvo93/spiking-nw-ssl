import random
import numpy as np
import torch


class SetSeed:
    def __init__(self, seed=42):
        """
        Set the random seed.
        """
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

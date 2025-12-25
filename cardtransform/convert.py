from typing import List, Tuple
import numpy as np
import random


class BaseLiner:
    """
    set baseline to 0, and add some noise to it.
    Args:
        ecg: shape: (n, lead_n, length) #n corresponds to the choose_num
    Returns:
        calibrated_ecg: shape: (n, lead_n, length)
    """
    def __init__(self, noise: Tuple=(0,0)):
      self.noise_range = noise

    def _get_baseline(self, leads: np.ndarray, bins=20) -> List[float]:
        """
        get baseline of ECG
        Args:
            leads: single sample of 12 leads ECG
                shape: (leads, length)
        Returns:
            baseline values of each lead
        """
        assert len(leads.shape) == 2
        baselines = []
        for lead in leads:
            try:
              counts, values = np.histogram(lead, bins)
            except ValueError:
              print(f"{lead=}")
              print(f"{bins=}")
              raise ValueError 
            baselines.append(values[np.argmax(counts)] + random.uniform(*self.noise_range))
        return baselines

    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        assert len(ecg.shape) == 3
        choose_num, lead_num, _ = ecg.shape
        ecg = ecg.reshape(choose_num*lead_num,-1)
        baselines = self._get_baseline(ecg)
        ecg = (ecg.T - baselines).T
        ecg = ecg.reshape(choose_num, lead_num, -1)
        return ecg

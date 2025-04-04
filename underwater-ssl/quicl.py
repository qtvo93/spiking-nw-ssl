import numpy as np
import scipy.optimize as opt

def thorp_absorption(f):
    """Compute absorption loss in dB/km for a given frequency f (Hz) using Thorp's empirical formula."""
    f_kHz = f / 1000  # Convert Hz to kHz
    alpha = (
        (0.11 * f_kHz**2) / (1 + f_kHz**2)
        + (44 * f_kHz**2) / (4100 + f_kHz**2)
        + 2.75e-4 * f_kHz**2
        + 0.003
    )  # dB/km
    return alpha / 1000  # Convert to dB/m

def estimate_distance(W, P, f):
    """Solve for d given source level W, received level P, and frequency f."""
    alpha = thorp_absorption(f)
    
    def loss_function(d):
        return W - 10 * np.log10(d) - alpha * d - P
    
    d_initial = 100  # Initial guess (meters)
    d_solution = opt.fsolve(loss_function, d_initial)[0]
    return d_solution

def estimate_W(set_number):
    """Estimate source level W based on tonal set number."""
    base_W = 158  # High Tonal Set (first set)
    return base_W - (set_number - 1) * 4

# Frequency sets
frequency_sets = [
    # [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388],
    # [52, 67, 82, 97, 115, 133, 151, 169, 204, 238, 286, 341, 391],
    # [55, 70, 85, 100, 118, 136, 154, 172, 207, 241, 289, 344, 394],
    # [58, 73, 88, 103, 121, 139, 157, 175, 210, 244, 292, 347, 397],
    # [61, 76, 91, 106, 124, 142, 160, 178, 213, 247, 295, 350, 400]
    [109, 127, 145, 163, 198, 232, 280, 335, 385]
]

distance_estimates = {}
P = 200  # Example received level in dB

for set_number, frequencies in enumerate(frequency_sets, start=1):
    W = estimate_W(set_number)
    distance_estimates[f"Set {set_number}"] = {f: estimate_distance(W, P, f) for f in frequencies}

print(distance_estimates)

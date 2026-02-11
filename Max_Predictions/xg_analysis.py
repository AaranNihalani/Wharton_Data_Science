# Packages
import numpy as np
import pandas as pd
import scipy.stats as stats


# Main
if __name__ == "__main__":
    # Load Games
    df_data = pd.read_csv("xg_toi.csv", header=None)
    print(f"Loaded {len(df_data)} records. \n\n")

    # Convert to numpy
    df_numpy = df_data.to_numpy(dtype=float)

    # Fitting Poisson distribution
    x_data = df_numpy[:,0]
    hist, bin_edges = np.histogram(x_data, bins=100)
    hist = hist / sum(hist)
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(hist)
    print("α: " + str(fit_alpha) + ", min(x): " + str(fit_loc) + ", β: " + str(fit_beta))

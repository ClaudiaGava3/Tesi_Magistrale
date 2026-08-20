import os
import numpy as np
import matplotlib.pyplot as plt

# Importa la tua funzione plot_histogram (o ricopiala qui sotto)
from scripts.main_copy import plot_histogram, ensure_clean_dir

# Percorsi
DATA_DIR = "data/"  # metti il tuo path corretto se diverso
PLOTS_DIR = "plots/"
hist_dir = os.path.join(PLOTS_DIR, "histograms_randB")
ensure_clean_dir(hist_dir)

robotic_system = "sth"

# Carica i nuovi dati generati
x_data = np.load(f"{DATA_DIR}{robotic_system}_x_vboc_randB.npy")
b_data = np.load(f"{DATA_DIR}{robotic_system}_b_vboc_randB.npy")
n_data = np.load(f"{DATA_DIR}{robotic_system}_n_horizons_vboc_randB.npy")
status_data = np.load(f"{DATA_DIR}{robotic_system}_status_vboc_randB.npy")

# 1. Istogramma Angoli e Velocità (indici 2 a 5)
plot_histogram(
    x_data[:, 2:6],
    title="Inputs_Angles_and_Velocities_randB",
    xlabel="Value",
    ylabel="Frequency",
    bins=50,
    saving_dir=hist_dir,
)

# 2. Istogramma Target Scaling
plot_histogram(
    b_data,
    title="Target_Scaling_Factor_randB",
    xlabel="Value [m]",
    ylabel="Frequency",
    bins=50,
    saving_dir=hist_dir,
)

# 3. Istogramma Orizzonti di Convergenza N
plot_histogram(
    n_data,
    title="Distribution_of_Converged_Horizons_N_randB",
    xlabel="Horizon Length (N steps)",
    ylabel="Frequency",
    bins=np.arange(19, 34, 2),
    saving_dir=hist_dir,
    xticks=np.arange(20, 33, 2),
)

print("Istogrammi generati con successo nella cartella:", hist_dir)
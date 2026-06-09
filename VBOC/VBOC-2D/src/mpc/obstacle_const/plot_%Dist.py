import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

def create_success_plot():
    # 1. Dati in ingresso
    # x = np.array([4, 6, 8, 10, 12])

    #Lidar distance
    x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

    # # not null x0
    # y1 = np.array([95.5, 87.2, 85, 82.2, 79.2])  # MPC + NN
    # y2 = np.array([95.8, 87.5, 34, 1.8, 0.2])   # MPC naive

    # null x0
    # y1 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # MPC + NN
    # y2 = np.array([100.0, 93.8, 62.0, 0.0, 0.0])   # MPC naive

    # Lidar distance with N=20
    y1 = np.array([10/13*100,9/13*100,5/13*100,3/13*100,3/13*100])  # MPC + NN
    y2 = np.array([5/13*100,4/13*100,2/13*100,2/13*100,2/13*100])   # MPC naive

    
    # 2. Interpolazione (per rendere le linee curve e morbide)
    # Creiamo un set di punti più denso (300 punti tra 4 e 12)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    spl1 = make_interp_spline(x, y1, k=2) # Grado 2 perché abbiamo solo 3 punti
    y1_smooth = spl1(x_smooth)
    
    spl2 = make_interp_spline(x, y2, k=2)
    y2_smooth = spl2(x_smooth)

    # 3. Creazione del Plot
    plt.figure(figsize=(10, 6))

    # Plot Caso 1: MPC + NN (Blu)
    plt.plot(x_smooth, y1_smooth, label='MPC + NN', color='royalblue', linewidth=2)
    plt.scatter(x, y1, color='royalblue', s=50) # Aggiungiamo i punti originali

    # Plot Caso 2: MPC naive (Rosso)
    plt.plot(x_smooth, y2_smooth, label='MPC naive', color='crimson', linewidth=2)
    plt.scatter(x, y2, color='crimson', s=50) # Aggiungiamo i punti originali

    # 4. Personalizzazione (Inglese)
    plt.title('Success rate vs LiDAR ray', fontweight='bold', pad=15)
    plt.suptitle('N = 20', fontsize=26, y=0.95) 
    plt.xlabel('LiDAR ray [m]')
    plt.ylabel('Success Rate [%]')
    
    # Range degli assi per chiarezza
    #plt.ylim(-5, 105)
    plt.xticks(x) # Mostra solo i valori 4, 8, 12 sull'asse X
    
    # Legenda in centro a destra
    plt.legend(loc='center right', frameon=True, shadow=True)

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_success_plot()
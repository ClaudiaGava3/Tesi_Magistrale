import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 32,     # Dimensione titolo
    'axes.labelsize': 28,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 26,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

def create_success_plot():
    # 1. Dati in ingresso
    x = np.array([4, 6, 8, 10, 12])

    #Lidar distance
    # x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    # x = np.array([1.5, 3.0, 4.5])
    # x = np.array([10,15,20,25,30])

    # # not null x0
    # y1 = np.array([95.5, 87.2, 85, 82.2, 79.2])  # MPC + NN
    # y2 = np.array([95.8, 87.5, 34, 1.8, 0.2])   # MPC naive

    # null x0
    y1 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # MPC + NN
    y2 = np.array([100.0, 93.8, 62.0, 0.0, 0.0])   # MPC naive

    # Lidar distance with N=20
    # y1 = np.array([10/13*100,9/13*100,5/13*100,3/13*100,3/13*100])  # MPC + NN
    # y2 = np.array([5/13*100,4/13*100,2/13*100,2/13*100,2/13*100])   # MPC naive

    # y1 = np.array([12/13*100,12/13*100,11/13*100])  #N10
    # y2 = np.array([11/13*100,9/13*100,3/13*100])   #N15
    # y3 = np.array([11/13*100,10/13*100,10/13*100])  #N20
    # y4 = np.array([11/13*100,10/13*100,11/13*100])   #N25
    # y5 = np.array([7/13*100,8/13*100,8/13*100])  #N30
    
    # y1 = np.array([0/13*100,0/13*100,0/13*100,0/13*100,0/13*100]) #LD1.5
    # y2 = np.array([0/13*100,2/13*100,1/13*100,1/13*100,1/13*100]) #LD3.0
    # y3 = np.array([0/13*100,9/13*100,1/13*100,1/13*100,2/13*100]) #LD4.5



    # # 2. Interpolazione (per rendere le linee curve e morbide)
    # # Creiamo un set di punti più denso (300 punti tra 4 e 12)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    spl1 = make_interp_spline(x, y1, k=1) # Grado 2 perché abbiamo solo 3 punti
    y1_smooth = spl1(x_smooth)
    # pchip1 = PchipInterpolator(x, y1)
    # y1_smooth = pchip1(x_smooth)
    
    spl2 = make_interp_spline(x, y2, k=1)
    y2_smooth = spl2(x_smooth)
    # pchip2 = PchipInterpolator(x, y2)
    # y2_smooth = pchip2(x_smooth)

    # spl3 = make_interp_spline(x, y3, k=2)
    # y3_smooth = spl3(x_smooth)

    # spl4 = make_interp_spline(x, y4, k=2)
    # y4_smooth = spl4(x_smooth)

    # spl5 = make_interp_spline(x, y5, k=2)
    # y5_smooth = spl5(x_smooth)



    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7))

    # Plot Caso 1: MPC + NN (Blu)
    plt.plot(x_smooth, y1_smooth, label='MPC+NN', color='royalblue', linewidth=2)
    plt.scatter(x, y1, color='royalblue', s=50) # Aggiungiamo i punti originali

    # Plot Caso 2: MPC naive (Rosso)
    plt.plot(x_smooth, y2_smooth, label='MPC naive', color='crimson', linewidth=2)
    plt.scatter(x, y2, color='crimson', s=50) # Aggiungiamo i punti originali

    # # Plot Caso 3: N20 (Verde)
    # plt.plot(x_smooth, y3_smooth, label='N=20', color='forestgreen', linewidth=2)
    # plt.scatter(x, y3, color='forestgreen', s=50) # Aggiungiamo i punti originali

    # # Plot Caso 4: N25 (Arancione)
    # plt.plot(x_smooth, y4_smooth, label='N=25', color='orange', linewidth=2)
    # plt.scatter(x, y4, color='orange', s=50) # Aggiungiamo i punti originali

    # # Plot Caso 5: N30 (Viola)
    # plt.plot(x_smooth, y5_smooth, label='N=30', color='purple', linewidth=2)
    # plt.scatter(x, y5, color='purple', s=50) # Aggiungiamo i punti originali

    # 4. Personalizzazione (Inglese)
    plt.title('Success rate vs Distance (N=15)', fontweight='bold', pad=15)
    plt.suptitle('Null x0', fontsize=26, y=0.95) 
    plt.xlabel('Distance [m]')
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
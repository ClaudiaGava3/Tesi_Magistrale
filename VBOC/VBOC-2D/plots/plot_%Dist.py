import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D

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
    # x = np.array([4, 6, 8, 10, 12])

    #Lidar distance
    # x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    # x = np.array([1.5, 3.0, 4.5])
    # x = np.array([10,15,20,25,30])

    x1 = np.array([0.115,0.69,1.33,1.66,1.94,2.375,5.56,10.575,18.005,29.275,44.655,57.935]) #FP% 1m
    y1 =  np.array([22.87,19.625,17.885,17.16,16.835,16.195,12.9,9.245,5.38,1.91,0.32,0.015]) #FN% 1m
    x2 = np.array([0.045,0.26,0.565,0.745,0.745,0.755,0.8,0.855,2.15,3.845,6.635,11.525,21.015,41.54]) #FP% 2.5m
    y2 = np.array([10.57,9.38,8.675,8.32,8.29,8.225,8.155,8.06,6.9,5.485,4.075,2.315,1.285,0.745]) #FN% 2.5m
    x3 = np.array([0.03,0.185,0.37,0.45,0.52,0.54,0.54,0.54,0.615,1.38,2.39,3.96,6.95,12.735,25.825]) #FP% 4m
    y3 = np.array([7.125,6.325,5.955,5.81,5.755,5.72,5.71,5.705,5.59,4.96,4.245,3.33,2.375,1.755,1.385]) #FN% 4m

    # Coordinate del punto di DeepReach (inserisci i tuoi valori calcolati)
    x_deepreach1 = 14.695
    y_deepreach1 = 16.835
    x_deepreach2_5 = 5.6
    y_deepreach2_5 = 8.29
    x_deepreach4 = 3.47
    y_deepreach4 = 5.705
    
   

    # # not null x0
    # y1 = np.array([95.5, 87.2, 85, 82.2, 79.2])  # MPC + NN
    # y2 = np.array([95.8, 87.5, 34, 1.8, 0.2])   # MPC naive

    # null x0
    # y1 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # MPC + NN
    # y2 = np.array([100.0, 93.8, 62.0, 0.0, 0.0])   # MPC naive

    # Lidar distance with N=20
    # y1 = np.array([10/13*100,9/13*100,5/13*100,3/13*100,3/13*100])  # MPC + NN
    # y2 = np.array([5/13*100,4/13*100,2/13*100,2/13*100,2/13*100])   # MPC naive

    # y1 = np.array([12/13*100,12/13*100,11/13*100])  #N10
    # y2 = np.array([11/13*100,9/13*100,3/13*100])   #N15
    # y3 = np.array([11/13*100,11/13*100,10/13*100])  #N20
    # y4 = np.array([11/13*100,10/13*100,11/13*100])   #N25
    # y5 = np.array([7/13*100,8/13*100,8/13*100])  #N30
    
    # y1 = np.array([3/13*100,6/13*100,0/13*100,0/13*100,0/13*100]) #LD1.5
    # y2 = np.array([2/13*100,8/13*100,5/13*100,6/13*100,1/13*100]) #LD3.0
    # y3 = np.array([3/13*100,1/13*100,3/13*100,6/13*100,3/13*100]) #LD4.5



    # # 2. Interpolazione (per rendere le linee curve e morbide)
    # # Creiamo un set di punti più denso (300 punti tra 4 e 12)
    # x_smooth = np.linspace(x.min(), x.max(), 300)
    
    # spl1 = make_interp_spline(x, y1, k=2) # Grado 2 perché abbiamo solo 3 punti
    # y1_smooth = spl1(x_smooth)
    # pchip1 = PchipInterpolator(x, y1)
    # y1_smooth = pchip1(x_smooth)
    
    # spl2 = make_interp_spline(x, y2, k=2)
    # y2_smooth = spl2(x_smooth)
    # pchip2 = PchipInterpolator(x, y2)
    # y2_smooth = pchip2(x_smooth)

    # spl3 = make_interp_spline(x, y3, k=2)
    # y3_smooth = spl3(x_smooth)
    # pchip3 = PchipInterpolator(x, y3)
    # y3_smooth = pchip3(x_smooth)

    # # spl4 = make_interp_spline(x, y4, k=2)
    # # # y4_smooth = spl4(x_smooth)
    # pchip4 = PchipInterpolator(x, y4)
    # y4_smooth = pchip4(x_smooth)

    # # # spl5 = make_interp_spline(x, y5, k=2)
    # # # y5_smooth = spl5(x_smooth)
    # pchip5 = PchipInterpolator(x, y5)
    # y5_smooth = pchip5(x_smooth)




    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7)) #15,8

    # # Plot Caso 1: MPC + NN (Blu)
    # plt.plot(x_smooth, y1_smooth, label='LD=1.5', color='royalblue', linewidth=2)
    # plt.scatter(x, y1, color='royalblue', s=50) # Aggiungiamo i punti originali

    # # Plot Caso 2: MPC naive (Rosso)
    # plt.plot(x_smooth, y2_smooth, label='LD=3.0', color='crimson', linewidth=2)
    # plt.scatter(x, y2, color='crimson', s=50) # Aggiungiamo i punti originali

    # # # Plot Caso 3: N20 (Verde)
    # plt.plot(x_smooth, y3_smooth, label='LD=4.5', color='forestgreen', linewidth=2)
    # plt.scatter(x, y3, color='forestgreen', s=50) # Aggiungiamo i punti originali

    # # # Plot Caso 4: N25 (Arancione)
    # plt.plot(x_smooth, y4_smooth, label='N=25', color='orange', linewidth=2)
    # plt.scatter(x, y4, color='orange', s=50) # Aggiungiamo i punti originali

    # # # Plot Caso 5: N30 (Viola)
    # plt.plot(x_smooth, y5_smooth, label='N=30', color='purple', linewidth=2)
    # plt.scatter(x, y5, color='purple', s=50) # Aggiungiamo i punti originali

    # Plot Caso 1: BoxMax=1.0m (Il marker='o' mette il puntino, il resto fa la linea continua)
    plt.plot(x1, y1, label='BoxMax=1.0m', color='royalblue', linewidth=2)
    
    # Plot Caso 2: BoxMax=2.5m
    plt.plot(x2, y2, label='BoxMax=2.5m', color='crimson', linewidth=2)
    
    # Plot Caso 3: BoxMax=4.0m
    plt.plot(x3, y3, label='BoxMax=4.0m', color='forestgreen', linewidth=2)

    # Plot Deepreach
    plt.scatter(x_deepreach1, y_deepreach1, 
                    color='grey',
                    marker='*',
                    s=500,
                    edgecolors='blue',
                    zorder=5)
    plt.scatter(x_deepreach2_5, y_deepreach2_5, 
                        color='grey',
                        marker='*',
                        s=500,
                        edgecolors='red',
                        zorder=5)
    plt.scatter(x_deepreach4, y_deepreach4, 
                        color='grey',
                        marker='*',
                        s=500,
                        edgecolors='green',
                        zorder=5)

    # # 4. Personalizzazione (Inglese)
    # plt.title('Failure rate vs horizon', fontweight='bold', pad=15)
    # # plt.suptitle('Null x0', fontsize=26, y=0.95) 
    # plt.xlabel('Horizon N')
    # plt.ylabel('Failure Rate [%]')

    plt.title('Trade-off FPR vs FNR', fontweight='bold', pad=15)
    plt.xlabel('FPR [%]')
    plt.ylabel('FNR [%]')
    
    # Range degli assi per chiarezza
    #plt.ylim(-5, 105)
    # plt.xticks(x) # Mostra solo i valori 4, 8, 12 sull'asse X
    
    # Prende le 3 linee appena disegnate (BoxMax 1, 2.5 e 4)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Crea la stella grigia "finta" per la legenda
    # linestyle='None' evita che venga disegnata una riga dietro la stella
    stella_unificata = Line2D([0], [0], linestyle='None', marker='*', markerfacecolor='gray', 
                              markeredgecolor='black', markersize=20, label='DeepReach')

    # Aggiunge la stella alla lista della legenda
    handles.append(stella_unificata)

    # Disegna la legenda finale con le 3 linee + 1 stella
    plt.legend(handles=handles, loc='upper right', frameon=True, shadow=True)

    # Legenda in centro a destra
    # plt.legend(loc='upper right', frameon=True, shadow=True)

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_success_plot()
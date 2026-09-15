import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 32,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 22,    # Dimensione numeri asse X
    'ytick.labelsize': 22,    # Dimensione numeri asse Y
    'legend.fontsize': 26,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

def create_tradeoffNN_plot():

    # Dati
    # 3D
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
    x_deepreach2_5_sec = 887*100/20000
    y_deepreach2_5_sec = 1498*100/20000
    x_deepreach4 = 3.47
    y_deepreach4 = 5.705

    # 2D
    x4 = np.array([55,179,409,794,1527,2755,4641,7537,12093,18238,19160])*100/20000 #FP%
    y4 = np.array([231,146,98,60,34,28,16,7,1,0,0])*100/20000 #FN%

    x_deepreach2_5_ = 199*100/20000
    y_deepreach2_5_ = 763*100/20000
   

    

    

    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7)) #15,8    

    # Plot Caso 1: BoxMax=1.0m (Il marker='o' mette il puntino, il resto fa la linea continua)
    # plt.plot(x1, y1, label='BoxMax=1.0m', color='royalblue', linewidth=2)
    
    # Plot Caso 2: BoxMax=2.5m
    plt.plot(x2, y2, label='Quadcopter', color='crimson', linewidth=2)
    plt.plot(x4, y4, label='Bicopter', color='blue', linewidth=2)
    
    # Plot Caso 3: BoxMax=4.0m
    # plt.plot(x3, y3, label='BoxMax=4.0m', color='forestgreen', linewidth=2)

    # Plot Deepreach
    # plt.scatter(x_deepreach1, y_deepreach1, 
    #                 color='grey',
    #                 marker='*',
    #                 s=500,
    #                 edgecolors='blue',
    #                 zorder=5)
    plt.scatter(x_deepreach2_5, y_deepreach2_5, 
                        color='grey',
                        marker='*',
                        s=500,
                        edgecolors='red',
                        zorder=5)
    plt.scatter(x_deepreach2_5_sec, y_deepreach2_5_sec, 
                            color='grey',
                            marker='*',
                            s=500,
                            edgecolors='orange',
                            zorder=5)
    plt.scatter(x_deepreach2_5_, y_deepreach2_5_, 
                            color='grey',
                            marker='*',
                            s=500,
                            edgecolors='blue',
                            zorder=5)
    # plt.scatter(x_deepreach4, y_deepreach4, 
    #                     color='grey',
    #                     marker='*',
    #                     s=500,
    #                     edgecolors='green',
    #                     zorder=5)

    plt.annotate(
      r'$\beta=0$',
      xy=(0,10),
      xytext=(10, 10),
      textcoords='offset points',
      fontsize=18)
    plt.annotate(
          r'$\beta=0$',
          xy=(0,1),
          xytext=(10, 10),
          textcoords='offset points',
          fontsize=18)
    plt.annotate(
              r'$\beta=1$',
              xy=(40,1),
              xytext=(10, 10),
              textcoords='offset points',
              fontsize=18)
    plt.annotate(
                  r'$\beta=1$',
                  xy=(90,0.3),
                  xytext=(10, 10),
                  textcoords='offset points',
                  fontsize=18)
    plt.annotate(
                      'first attempt',
                      xy=(8,8),
                      xytext=(10, 10),
                      textcoords='offset points',
                      fontsize=18)
    plt.annotate(
                      'second attempt',
                      xy=(7,7),
                      xytext=(10, 10),
                      textcoords='offset points',
                      fontsize=18)

    # # 4. Personalizzazione (Inglese)
    # plt.title('Trade-off for NN', fontweight='bold', pad=15)
    plt.xlabel('False alarm rate [%]')
    plt.ylabel('Missed alarm rate [%]')
    
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

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

def create_tradeoffDEEP_plot():

    # Dati
    # 2D
    x2 = np.array([199,1891,3185,3653,3699,4466,6878,9033,10848,12496,13870,15001,15973,16752])*100/20000 #FP% 2.5m
    y2 = np.array([763,405,274,231,228,165,57,26,15,7,2,1,1,1])*100/20000 #FN% 2.5m

    # Coordinate del punto di DeepReach (inserisci i tuoi valori calcolati)
    x_NN2_5 = 55*100/20000
    y_NN2_5 = 231*100/20000

    # 3D
    x4 = np.array([1120,1574,2082,2700,3327,4028,4733,5491,6289,7093,7916])*100/20000 #FP%
    y4 = np.array([1658,1362,1101,856,676,533,441,348,295,252,224])*100/20000 #FN%
    x5 = np.array([887,1316,1807,2360,2988,3645,4333,5090,5864,6562,7338])*100/20000 #FP%
    y5 = np.array([1498,1162,914,724,576,442,359,292,245,213,181])*100/20000 #FN%
    

    x_NN2_5_ = 9*100/20000
    y_NN2_5_ = 2114*100/20000
   

    

    

    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7)) #15,8    

    # Plot Caso 1: BoxMax=1.0m (Il marker='o' mette il puntino, il resto fa la linea continua)
    # plt.plot(x1, y1, label='BoxMax=1.0m', color='royalblue', linewidth=2)
    
    # Plot Caso 2: BoxMax=2.5m
    plt.plot(x4, y4, label=r'Quadcopter $1^{th}$attempt', color='crimson', linewidth=2)
    plt.plot(x5, y5, label=r'Quadcopter $2^{nd}$attempt', color='orange', linewidth=2)
    plt.plot(x2, y2, label='Bicopter', color='blue', linewidth=2)
    
    # Plot Caso 3: BoxMax=4.0m
    # plt.plot(x3, y3, label='BoxMax=4.0m', color='forestgreen', linewidth=2)

    # Plot Deepreach
    # plt.scatter(x_deepreach1, y_deepreach1, 
    #                 color='grey',
    #                 marker='*',
    #                 s=500,
    #                 edgecolors='blue',
    #                 zorder=5)
    plt.scatter(x_NN2_5, y_NN2_5, 
                        color='grey',
                        marker='*',
                        s=500,
                        edgecolors='red',
                        zorder=5)
    plt.scatter(x_NN2_5_, y_NN2_5_, 
                            color='grey',
                            marker='*',
                            s=500,
                            edgecolors='blue',
                            zorder=5)
    # plt.scatter(x_deepreach4, y_deepreach4, 
    #                     color='grey',
    #                     marker='*',
    #                     s=500,
    #                     edgecolors='green',
    #                     zorder=5)

    plt.annotate(
      r'$\alpha=0$',
      xy=(0,3.8),
      xytext=(10, 10),
      textcoords='offset points',
      fontsize=18)
    plt.annotate(
          r'$\alpha=0$',
          xy=(7,8),
          xytext=(10, 10),
          textcoords='offset points',
          fontsize=18)
    plt.annotate(
              r'$\alpha=1$',
              xy=(40,1),
              xytext=(10, 10),
              textcoords='offset points',
              fontsize=18)
    plt.annotate(
                  r'$\alpha=1$',
                  xy=(80,0.2),
                  xytext=(10, 10),
                  textcoords='offset points',
                  fontsize=18)

    # # 4. Personalizzazione (Inglese)
    # plt.title('Trade-off for DeepReach', fontweight='bold', pad=15)
    plt.xlabel('False alarm rate [%]')
    plt.ylabel('Missed alarm rate [%]')
    
    # Prende le 3 linee appena disegnate (BoxMax 1, 2.5 e 4)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Crea la stella grigia "finta" per la legenda
    # linestyle='None' evita che venga disegnata una riga dietro la stella
    stella_unificata = Line2D([0], [0], linestyle='None', marker='*', markerfacecolor='gray', 
                              markeredgecolor='black', markersize=20, label='NN')

    # Aggiunge la stella alla lista della legenda
    handles.append(stella_unificata)

    # Disegna la legenda finale con le 3 linee + 1 stella
    plt.legend(handles=handles, loc='upper right', frameon=True, shadow=True)

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

def create_tradeoffBICOPTER_plot():

    # Dati
    # DEEP
    x2 = np.array([199,1891,3185,3653,3699,4466,6878,9033,10848,12496,13870,15001,15973,16752])*100/20000 #FP% 2.5m
    y2 = np.array([763,405,274,231,228,165,57,26,15,7,2,1,1,1])*100/20000 #FN% 2.5m

    # NN
    x4 = np.array([55,179,409,794,1527,2755,4641,7537,12093,18238,19160])*100/20000 #FP%
    y4 = np.array([231,146,98,60,34,28,16,7,1,0,0])*100/20000 #FN%
    
    
  

    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7)) #15,8    

    # Plot Caso 1: BoxMax=1.0m (Il marker='o' mette il puntino, il resto fa la linea continua)
    # plt.plot(x1, y1, label='BoxMax=1.0m', color='royalblue', linewidth=2)
    
    # Plot Caso 2: BoxMax=2.5m
    plt.plot(x4, y4, label='Neural Network', color='crimson', linewidth=2)
    plt.plot(x2, y2, label='DeepReach', color='blue', linewidth=2)
    
    
    plt.annotate(
      r'$\alpha=0$',
      xy=(1.0,3.6),
      xytext=(10, 10),
      textcoords='offset points',
      fontsize=18)
    plt.annotate(
          r'$\beta=0$',
          xy=(1.0,1.0),
          xytext=(10, 10),
          textcoords='offset points',
          fontsize=18)
    plt.annotate(
              r'$\alpha=1$',
              xy=(80,0.0),
              xytext=(10, 10),
              textcoords='offset points',
              fontsize=18)
    plt.annotate(
                  r'$\beta=1$',
                  xy=(80,0.2),
                  xytext=(10, 10),
                  textcoords='offset points',
                  fontsize=18)

    # # 4. Personalizzazione (Inglese)
    # plt.title('Trade-off for DeepReach', fontweight='bold', pad=15)
    plt.xlabel('False alarm rate [%]')
    plt.ylabel('Missed alarm rate [%]')
    
    # Prende le 3 linee appena disegnate (BoxMax 1, 2.5 e 4)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Crea la stella grigia "finta" per la legenda
    # linestyle='None' evita che venga disegnata una riga dietro la stella
    stella_unificata = Line2D([0], [0], linestyle='None', marker='*', markerfacecolor='gray', 
                              markeredgecolor='black', markersize=20, label='NN')

    # Aggiunge la stella alla lista della legenda
    # handles.append(stella_unificata)

    # Disegna la legenda finale con le 3 linee + 1 stella
    plt.legend(handles=handles, loc='upper right', frameon=True, shadow=True)

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

def create_tradeoffQUADCOPTER_plot():

    # Dati
    # DEEP
    x4 = np.array([1120,1574,2082,2700,3327,4028,4733,5491,6289,7093,7916])*100/20000 #FP%
    y4 = np.array([1658,1362,1101,856,676,533,441,348,295,252,224])*100/20000 #FN%
    x5 = np.array([887,1316,1807,2360,2988,3645,4333,5090,5864,6562,7338])*100/20000 #FP%
    y5 = np.array([1498,1162,914,724,576,442,359,292,245,213,181])*100/20000 #FN%

    # NN
    x2 = np.array([0.045,0.26,0.565,0.745,0.745,0.755,0.8,0.855,2.15,3.845,6.635,11.525,21.015,41.54]) #FP% 2.5m
    y2 = np.array([10.57,9.38,8.675,8.32,8.29,8.225,8.155,8.06,6.9,5.485,4.075,2.315,1.285,0.745]) #FN% 2.5m
    
    
  

    # 3. Creazione del Plot
    plt.figure(figsize=(12, 7)) #15,8    

    # Plot Caso 1: BoxMax=1.0m (Il marker='o' mette il puntino, il resto fa la linea continua)
    # plt.plot(x1, y1, label='BoxMax=1.0m', color='royalblue', linewidth=2)
    
    # Plot Caso 2: BoxMax=2.5m
    plt.plot(x2, y2, label='Neural Network', color='crimson', linewidth=2)
    plt.plot(x4, y4, label=r'DeepReach $1^{th}$attempt', color='blue', linewidth=2)
    plt.plot(x5, y5, label=r'DeepReach $2^{th}$attempt', color='turquoise', linewidth=2)
    
    
    plt.annotate(
      r'$\beta=0$',
      xy=(0.6,10.0),
      xytext=(10, 10),
      textcoords='offset points',
      fontsize=18)
    plt.annotate(
          r'$\alpha=0$',
          xy=(2.0,7.5),
          xytext=(10, 10),
          textcoords='offset points',
          fontsize=18)
    plt.annotate(
              r'$\beta=1$',
              xy=(39,1.3),
              xytext=(10, 10),
              textcoords='offset points',
              fontsize=18)
    plt.annotate(
                  r'$\alpha=1$',
                  xy=(38,1.8),
                  xytext=(10, 10),
                  textcoords='offset points',
                  fontsize=18)

    # # 4. Personalizzazione (Inglese)
    # plt.title('Trade-off for DeepReach', fontweight='bold', pad=15)
    plt.xlabel('False alarm rate [%]')
    plt.ylabel('Missed alarm rate [%]')
    
    # Prende le 3 linee appena disegnate (BoxMax 1, 2.5 e 4)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Crea la stella grigia "finta" per la legenda
    # linestyle='None' evita che venga disegnata una riga dietro la stella
    stella_unificata = Line2D([0], [0], linestyle='None', marker='*', markerfacecolor='gray', 
                              markeredgecolor='black', markersize=20, label='NN')

    # Aggiunge la stella alla lista della legenda
    # handles.append(stella_unificata)

    # Disegna la legenda finale con le 3 linee + 1 stella
    plt.legend(handles=handles, loc='upper right', frameon=True, shadow=True)

    # Griglia e stile
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Visualizzazione
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_tradeoffQUADCOPTER_plot()
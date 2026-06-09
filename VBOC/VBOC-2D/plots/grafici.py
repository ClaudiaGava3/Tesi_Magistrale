import matplotlib.pyplot as plt

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

# ===== DATI (modifica qui le ampiezze degli spicchi) =====
# #x0=0
# grafico1 = [100, 0]
# grafico2 = [62, 38]
# grafico3 = [100, 0]

# #x0!=0
# grafico4 = [85, 15]
# grafico5 = [34, 66]
# grafico6 = [100, 0]

grafico1 = [8/13, 5/13]
grafico2 = [7/13, 5/13]
grafico3 = [9/13, 4/13]
grafico4 = [6/13, 7/13]


grafico5 = [1/13, 12/13]
grafico6 = [3/13, 10/13]
grafico7 = [5/13, 8/13]
grafico8 = [4/13, 9/13]

# Etichette comuni
# etichette = ['Success','Impact']
etichette = ['Success','Failure']


# Colori uguali per tutti i grafici
colori = [ '#66b3ff','#ff9999']

# Titoli dei grafici
# titoli = ['MPC-NN (N=15)', 'MPC naive (N=15)', 'MPC naive (N=30)', 'MPC-NN (N=15)', 'MPC naive (N=15)', 'MPC naive (N=30)']
titoli = ['MPC-NN (N=10)', 'MPC-NN (N=15)', 'MPC-NN (N=20)', 'MPC-NN (N=30)', 'MPC naive (N=10)', 'MPC naive (N=15)', 'MPC naive (N=20)', 'MPC naive (N=30)']

# ===== CREAZIONE FIGURA =====
# Aumentiamo l'altezza a 12 per dare respiro ai titoli
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

dati = [grafico1, grafico2, grafico3, grafico4, grafico5, grafico6, grafico7, grafico8]

# ===== CREAZIONE GRAFICI =====
for i, ax in enumerate(axs.flat):
    # Riduciamo leggermente il font delle percentuali (es. 18) per non farle uscire
    ax.pie(dati[i],
           labels=None,
           colors=colori,
           autopct='%1.1f%%',
           startangle=90,
           textprops={'fontsize': 20, 'weight': 'bold', 'color': 'white'})
    
    ax.set_title(titoli[i], fontsize=22, pad=20) # 'pad' distanzia il titolo dal cerchio

# ===== TITOLI DI RIGA =====
# Usiamo 'y' più precisi e fontweight per la chiarezza
# fig.text(0.5, 0.95, 'Null $x_0$', ha='center', fontsize=28)
# fig.text(0.5, 0.49, 'Not Null $x_0$', ha='center', fontsize=28)
fig.text(0.5, 0.95, 'MPC-NN (LiDAR ray = 1.5m)', ha='center', fontsize=28)
fig.text(0.5, 0.49, 'MPC naive (LiDAR ray = 1.5m)', ha='center', fontsize=28)

# ===== LEGENDA ESTERNA =====
# La mettiamo in alto al centro o la spostiamo leggermente per non coprire i titoli
fig.legend(etichette, loc='upper center', bbox_to_anchor=(0.78, 1.0), 
           ncol=2, fontsize=24, frameon=True)

# ===== CONTROLLO LAYOUT =====
# 1. Applichiamo tight_layout per organizzare gli elementi base
# rect=[sinistra, basso, destra, alto] lascia spazio in alto per il titolo
plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 

# 2. Usiamo hspace per distanziare le righe (0.5 è un buon punto di partenza, aumenta se serve)
plt.subplots_adjust(hspace=0.6) 

# Se vuoi che il titolo "Not Null x0" sia esattamente al centro dello spazio creato:
# fig.text(0.5, 0.48, ...) va bene, ma se sposti hspace potresti doverlo regolare a 0.50

plt.show()
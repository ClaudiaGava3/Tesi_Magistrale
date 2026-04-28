import matplotlib.pyplot as plt

# ===== DATI (modifica qui le ampiezze degli spicchi) =====
grafico1 = [95, 5]
grafico2 = [40.2,59.8,]
grafico3 = [100,0]

# Etichette comuni
etichette = ['Success','Impact']

# Colori uguali per tutti i grafici
colori = [ '#66b3ff','#ff9999']

# Titoli dei grafici
titoli = ['MPC-NN (N=10)', 'MPC (N=10)', 'MPC (N=50)']

# ===== CREAZIONE FIGURA =====
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# Lista dati per iterazione
dati = [grafico1, grafico2, grafico3]

# ===== CREAZIONE GRAFICI =====
for i, ax in enumerate(axs):
    ax.pie(dati[i],
           labels=None,  # niente etichette sui singoli grafici
           colors=colori,
           autopct='%10.1f%%',
           startangle=90)
    
    ax.set_title(titoli[i], fontsize=18)

# ===== LEGENDA UNICA (in alto a destra) =====
fig.legend(etichette,
           loc='upper right')

# Migliora layout
plt.tight_layout()

# Mostra grafici
plt.show()
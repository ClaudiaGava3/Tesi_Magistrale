import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Importa le tue librerie
from parser import Parameters
from mpc_abstract import Model
from mpc_controller import MpcController

# sempre 400 test

# orizzonte breve N=15

# velocità tra +-1 angoli tra +-pi/4
# ostacolo a 4m: NN->95.5, naive->95.8
# ostacolo a 6m: NN->87.2, naive->87.5
# ostacolo a 8m: NN->85, naive->34
# ostacolo a 10m: NN->82.2, naive->1.8
# ostacolo a 12m: NN->79.2, naive->0.2

# velocità e angoli nulli
# ostacolo a 4m: NN-> 100, naive->100
# ostacolo a 6m: NN-> 100, naive->93.8
# ostacolo a 8m: NN-> 100, naive->62.0
# ostacolo a 10m: NN->100, naive->0.0
# ostacolo a 12m: NN->100, naive->3.2


# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

def genera_condizioni_iniziali(x_muro_fisso):
    """
    Genera stati iniziali e posizioni target randomiche 
    seguendo esattamente la definizione del Capitolo 5.2.2.
    """
    # 1. Posizione iniziale fissa (origine)
    x0_pos = 0.0
    z0_pos = 0.0 
    
    # 2. Angoli e velocità randomiche
    theta0 = np.random.uniform(-np.pi/4, np.pi/4) 
    vx0 = np.random.uniform(-1.0, 1.0) 
    vz0 = np.random.uniform(-1.0, 1.0)
    wy0 = np.random.uniform(-1.0, 1.0)
    
    x0 = np.array([x0_pos, z0_pos, theta0, vx0, vz0, wy0])
    
    # 3. Delta Ostacolo randomico tra 0.2 e 0.6 metri
    delta_obs = np.random.uniform(0.2, 0.6)
    
    # 4. Posizione del Target: appena prima dell'ostacolo
    xref_pos = x_muro_fisso - delta_obs
    zref_pos = 0.0  # Volo in linea retta
    
    x_ref = np.array([xref_pos, zref_pos, 0.0, 0.0, 0.0, 0.0])
    
    return x0, x_ref, delta_obs

def main():
    print("--- Inizializzazione Campagna Statistica 1000 Test ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' 
    params.build = True

    model = Model(params)
    controller = MpcController(model)

    # --- PARAMETRI SIMULAZIONE ---
    NUM_TESTS = 400
    DT = params.dt
    SIM_TIME = 3.5
    N_SIM = int(SIM_TIME / DT)
    
    TOLLERANZA_TARGET = 0.1  # Errore massimo accettabile dal target
    X_MURO_FISSO = 8.0       # Ostacolo sufficientemente lontano

    # Contatori statistici
    arrivati = 0
    impattati = 0
    falliti_matematicamente = 0

    # Liste per salvare i voli di successo (per i plot)
    voli_successo = []

    print(f"\nAvvio {NUM_TESTS} simulazioni. Volo: {SIM_TIME}s, Ostacolo a: {X_MURO_FISSO}m")
    start_total = time.time()

    for i in range(NUM_TESTS):
        x0, x_ref, delta_obs = genera_condizioni_iniziali(X_MURO_FISSO)
        current_x = x0.copy()
        
        solver_fallito = False
        impatto_fisico = False
        
        # Storia locale per l'eventuale plot
        x_hist = [current_x]
        
        # Warm start
        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        x_guess_safe = np.array([current_x[0], current_x[1], 0.0, 0.0, 0.0, 0.0])
        
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x_guess_safe, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # Loop di simulazione del singolo volo
        for t in range(N_SIM):
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, X_MURO_FISSO)
            
            # Controllo impatto fisico reale
            if current_x[0] >= X_MURO_FISSO:
                impatto_fisico = True
                break
                
            # Controllo fallimento matematico dell'MPC (Status 4)
            if status not in [0, 2]:
                solver_fallito = True
                break
                
            current_x = x_sol[1]
            x_hist.append(current_x)

        # --- VALUTAZIONE FINALE DEL VOLO ---
        # Se ha impattato o ha superato il muro all'ultimo istante
        if impatto_fisico or current_x[0] >= X_MURO_FISSO:
            impattati += 1
            esito = "Impatto"
        # Se l'ottimizzatore è andato in panico (Miopia del Naive)
        elif solver_fallito:
            falliti_matematicamente += 1
            esito = "Infeasible (Status 4)"
        # Se è sopravvissuto, controlliamo se è arrivato al target
        else:
            errore_pos = np.abs(current_x[0] - x_ref[0])
            if errore_pos <= TOLLERANZA_TARGET:
                arrivati += 1
                esito = "Successo"
                
                # Salviamo i dati per i plot (massimo 10)
                if len(voli_successo) < 10:
                    voli_successo.append({
                        "id": i + 1,
                        "x_hist": np.array(x_hist),
                        "xref": x_ref,
                        "delta_obs": delta_obs
                    })
            else:
                # Caso raro: il tempo è scaduto prima che arrivasse
                falliti_matematicamente += 1 

        # Barra di progresso
        percentuale = (i + 1) / NUM_TESTS
        barra = '█' * int(30 * percentuale) + '-' * (30 - int(30 * percentuale))
        sys.stdout.write(f"\rProgresso: [{barra}] {percentuale*100:.1f}% ({i+1}/{NUM_TESTS})")
        sys.stdout.flush()

    end_total = time.time()
    print(f"\n\nCampagna completata in {(end_total - start_total)/60:.1f} minuti.")

    # --- RESOCONTO STATISTICO ---
    print("\n================================================")
    print("           RESOCONTO STATISTICO FINALE          ")
    print("================================================")
    print(f"Test Totali             : {NUM_TESTS}")
    print(f"Successi (A Target)     : {arrivati} ({(arrivati/NUM_TESTS)*100:.1f}%)")
    print(f"Impatti Fisici contro ostacolo : {impattati} ({(impattati/NUM_TESTS)*100:.1f}%)")
    print(f"Fallimenti (Infeasible/Status4): {falliti_matematicamente} ({(falliti_matematicamente/NUM_TESTS)*100:.1f}%)")
    print("================================================\n")

    # --- GENERAZIONE PLOT DEI 10 TEST RIUSCITI ---
    if len(voli_successo) > 0:
        cartella_plots = os.path.join("plots", "statistica")
        os.makedirs(cartella_plots, exist_ok=True)
        print(f"Salvataggio plot di {len(voli_successo)} voli di successo in: '{cartella_plots}'")

        plt.figure(figsize=(12, 6))
        
        # Disegno del muro fisso a destra
        plt.axvspan(X_MURO_FISSO, X_MURO_FISSO + 1.0, color='gray', alpha=0.5, label='Ostacolo')

        colors = plt.cm.tab10(np.linspace(0, 1, len(voli_successo)))

        for res, color in zip(voli_successo, colors):
            x_h = res['x_hist']
            xref = res['xref']
            
            # Traiettoria nel piano X-Z
            plt.plot(x_h[:, 0], x_h[:, 1], color=color, linewidth=2, alpha=0.8)
            # Segnamo il target per questa specifica traiettoria
            plt.scatter(xref[0], xref[1], color=color, marker='X', s=100, zorder=5)

        plt.title(f'Trajectory of {len(voli_successo)} test with Success')
        plt.xlabel('Position X [m]')
        plt.ylabel('Position Z [m]')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Etichette generiche alla legenda
        plt.plot([], [], color='black', linewidth=2, label='Trajectory Drone')
        plt.scatter([], [], color='black', marker='X', s=100, label='Target')
        #plt.legend(loc='lower left')

        plt.savefig(os.path.join(cartella_plots, "10_test_successo.png"), dpi=300)
        plt.show()
    else:
        print("Nessun volo di successo da plottare. (Questo è normale se stai testando il Naive MPC!)")

if __name__ == '__main__':
    main()
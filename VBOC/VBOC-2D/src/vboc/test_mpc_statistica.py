import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork

from parser import Parameters
from mpc_abstract import Model
from mpc_controller import MpcController

def genera_condizioni_ragionevoli():
    """Genera stati fisicamente possibili per non far impazzire il solver."""
    distanza_accettabile = False
    
    while not distanza_accettabile:
        # Partenza (Leggermente perturbata, ma gestibile in 0.4s di orizzonte)
        x0_pos = np.random.uniform(0.0, 0.0)
        z0_pos = np.random.uniform(0.0, 0.0) 
        theta0 = np.random.uniform(-np.radians(15), np.radians(15)) 
        vx0 = np.random.uniform(-1.0, 1.0) # Velocità di partenza gestibili
        vz0 = np.random.uniform(-1.0, 1.0)
        wy0 = np.random.uniform(-1.0, 1.0)
        
        # Target (Sempre FERMO e DRITTO)
        xref_pos = np.random.uniform(-3.0, 3.0)
        zref_pos = np.random.uniform(-3.0, 3.0)
        
        distanza = np.sqrt((xref_pos - x0_pos)**2 + (zref_pos - z0_pos)**2)
        
        # Accettiamo solo distanze coperte agevolmente in 5-10 secondi
        if 0.5 <= distanza <= 3.5: 
            distanza_accettabile = True
            
    x0 = np.array([x0_pos, z0_pos, theta0, vx0, vz0, wy0])
    x_ref = np.array([xref_pos, zref_pos, 0.0, 0.0, 0.0, 0.0]) # Target fermo
    
    return x0, x_ref

def main():
    print("--- Inizializzazione Campagna Test ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' 
    params.build = False 

    model = Model(params)
    controller = MpcController(model)

    # --- PARAMETRI CAMPAGNA ---
    NUM_TESTS = 100
    DT = params.dt
    SIM_TIME = 8.0  # Aumentato a 8 secondi per dare tempo al drone
    N_SIM = int(SIM_TIME / DT)
    ALPHA_REAL = 2.5 
    TOLLERANZA_TARGET = 0.05 # Entro 5 cm è successo

    risultati_errore = []
    risultati_max_alpha = []
    risultati_status = []

    print(f"\nAvvio {NUM_TESTS} simulazioni. Volo: {SIM_TIME}s")
    start_total = time.time()

    for i in range(NUM_TESTS):
        x0, x_ref = genera_condizioni_ragionevoli()
        current_x = x0.copy()
        
        max_alpha_test = 0.0
        solver_fallito = False
        
        # =========================================================
        # WARM-START INTELLIGENTE (Salva la vita al solver)
        # Anche se parte inclinato, il "guess" è dritto e in hovering.
        # =========================================================
        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        x_guess_safe = np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0])
        
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x_guess_safe, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # Volo
        for t in range(N_SIM):
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, ALPHA_REAL)
            
            if status not in [0, 2]:
                solver_fallito = True
                break
                
            if alpha_curr > max_alpha_test:
                max_alpha_test = alpha_curr
                
            current_x = x_sol[1]

        # Valutazione
        if solver_fallito:
            risultati_status.append("Errore Solver")
            risultati_errore.append(np.nan)
            risultati_max_alpha.append(max_alpha_test)
        else:
            errore_pos = np.linalg.norm(current_x[:2] - x_ref[:2])
            risultati_errore.append(errore_pos)
            risultati_max_alpha.append(max_alpha_test)
            
            if errore_pos <= TOLLERANZA_TARGET:
                risultati_status.append("Arrivato")
            else:
                risultati_status.append("Fuori Tolleranza")

        # Barra progressi
        percentuale = (i + 1) / NUM_TESTS
        lunghezza_barra = 30
        riempimento = int(lunghezza_barra * percentuale)
        barra = '█' * riempimento + '-' * (lunghezza_barra - riempimento)
        sys.stdout.write(f"\rProgresso: [{barra}] {percentuale*100:.1f}% ({i+1}/{NUM_TESTS})")
        sys.stdout.flush()

    end_total = time.time()
    print(f"\n\nCampagna completata in {(end_total - start_total)/60:.1f} minuti.")

    # --- ANALISI E PLOT ---
    err_array = np.array(risultati_errore)
    alpha_array = np.array(risultati_max_alpha)
    status_array = np.array(risultati_status)

    arrivati = np.sum(status_array == "Arrivato")
    fuori_toll = np.sum(status_array == "Fuori Tolleranza")
    errori = np.sum(status_array == "Errore Solver")
    
    violazioni_sicurezza = np.sum(alpha_array > ALPHA_REAL)

    print("\n--- RESOCONTO STATISTICO ---")
    print(f"Successi (entro {TOLLERANZA_TARGET}m): {arrivati} ({(arrivati/NUM_TESTS)*100:.1f}%)")
    print(f"Lenti/Fuori Tolleranza      : {fuori_toll} ({(fuori_toll/NUM_TESTS)*100:.1f}%)")
    print(f"Errori (Solver Crash)       : {errori} ({(errori/NUM_TESTS)*100:.1f}%)")
    print(f"VIOLAZIONI SICUREZZA ALPHA  : {violazioni_sicurezza} test su {NUM_TESTS}")

    # Plot Scatter
    plt.figure(figsize=(9, 6))
    valid_idx = ~np.isnan(err_array)
    colori = ['blue' if e <= TOLLERANZA_TARGET else 'orange' for e in err_array[valid_idx]]
    
    plt.scatter(err_array[valid_idx], alpha_array[valid_idx], alpha=0.6, c=colori, edgecolor='k')
    plt.axvline(TOLLERANZA_TARGET, color='green', linestyle='--', label=f'Tolleranza Target ({TOLLERANZA_TARGET}m)')
    plt.axhline(ALPHA_REAL, color='red', linestyle='--', label=f'Limite Alpha Reale ({ALPHA_REAL}m)')
    
    plt.title("Risultati 100 Test: Sicurezza (Alpha) vs Accuratezza (Errore)")
    plt.xlabel("Errore di Posizione Finale dal Target [m]")
    plt.ylabel("Picco Massimo di Alpha Richiesto [m]")
    
    plt.text(TOLLERANZA_TARGET/2, ALPHA_REAL - 0.2, 'OTTIMI E SICURI', color='green', weight='bold', ha='center')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import random
from learning import NeuralNetwork

# Importa le tue librerie
from parser import Parameters
from mpc_abstract import Model
from mpc_controller import MpcController

def genera_condizioni_iniziali():
    """Genera stati fisicamente possibili per non far impazzire il solver."""
    distanza_accettabile = False
    
    while not distanza_accettabile:
        # Partenza da (0,0) ma con angoli e velocità randomici
        x0_pos = 0.0
        z0_pos = 0.0 
        theta0 = np.random.uniform(-np.radians(45), np.radians(45)) 
        vx0 = np.random.uniform(-1.0, 1.0) 
        vz0 = np.random.uniform(-1.0, 1.0)
        wy0 = np.random.uniform(-1.0, 1.0)
        
        # Target casuale tra -3 e 3 metri
        xref_pos = np.random.uniform(-5.0, 5.0)
        zref_pos = np.random.uniform(-5.0, 5.0)
        
        distanza = np.sqrt((xref_pos - x0_pos)**2 + (zref_pos - z0_pos)**2)
        
        # Solo distanze sensate nel tempo di simulazione dato
        if 0.5 <= distanza <= 5.0: 
            distanza_accettabile = True
            
    x0 = np.array([x0_pos, z0_pos, theta0, vx0, vz0, wy0])
    x_ref = np.array([xref_pos, zref_pos, 0.0, 0.0, 0.0, 0.0])
    
    return x0, x_ref

def main():
    print("--- Inizializzazione Campagna Test ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' 
    params.build = True 

    model = Model(params)
    controller = MpcController(model)

    # --- PARAMETRI SIMULAZIONE ---
    NUM_TESTS = 1000
    DT = params.dt
    SIM_TIME = 3.0 
    N_SIM = int(SIM_TIME / DT)
    ALPHA_REAL = 1.0 
    TOLLERANZA_TARGET = 0.05 

    # Inizializzazione liste
    risultati_errore = []
    risultati_max_alpha = []
    risultati_status = []
    voli_completati = []

    print(f"\nAvvio {NUM_TESTS} simulazioni. Volo: {SIM_TIME}s")
    start_total = time.time()

    for i in range(NUM_TESTS):
        x0, x_ref = genera_condizioni_iniziali()
        current_x = x0.copy()
        
        max_alpha_test = 0.0
        solver_fallito = False
        
        # Storia locale per il singolo volo
        x_hist = [current_x]
        u_hist = []
        alpha_hist = []
        
        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        x_guess_safe = np.array([x0[0], x0[1], 0.0, 0.0, 0.0, 0.0])
        
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x_guess_safe, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # Loop di simulazione
        for t in range(N_SIM):
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, ALPHA_REAL)
            
            if status not in [0, 2]:
                solver_fallito = True
                break
                
            if alpha_curr > max_alpha_test:
                max_alpha_test = alpha_curr
                
            current_x = x_sol[1]
            x_hist.append(current_x)
            u_hist.append(u_sol[0])
            alpha_hist.append(alpha_curr)

       # Valutazione finale del test
        if solver_fallito:
            risultati_status.append("Errore Solver")
            risultati_errore.append(np.nan)
            risultati_max_alpha.append(max_alpha_test)
        else:
            errore_pos = np.linalg.norm(current_x[:2] - x_ref[:2])
            risultati_errore.append(errore_pos)
            risultati_max_alpha.append(max_alpha_test)
            
            # --- NUOVA LOGICA DI VALUTAZIONE FISICA REALE ---
            if max_alpha_test >= ALPHA_REAL:
                esito = "Impatto"
            elif errore_pos <= TOLLERANZA_TARGET:
                esito = "Arrivato"
            else:
                esito = "Fuori Tolleranza"
                
            risultati_status.append(esito)
            
            # Salvataggio dati (salviamo tutto, il filtro per i plot lo facciamo dopo)
            voli_completati.append({
                "id": i + 1,
                "x_hist": np.array(x_hist),
                "u_hist": np.array(u_hist),
                "alpha_hist": np.array(alpha_hist),
                "xref": x_ref,
                "status": esito
            })

        # Barra di progresso
        percentuale = (i + 1) / NUM_TESTS
        barra = '█' * int(30 * percentuale) + '-' * (30 - int(30 * percentuale))
        sys.stdout.write(f"\rProgresso: [{barra}] {percentuale*100:.1f}% ({i+1}/{NUM_TESTS})")
        sys.stdout.flush()

    end_total = time.time()
    print(f"\n\nCampagna completata in {(end_total - start_total)/60:.1f} minuti.")

    # --- ANALISI STATISTICA ---
    err_array = np.array(risultati_errore)
    alpha_array = np.array(risultati_max_alpha)
    status_array = np.array(risultati_status)

    arrivati = np.sum(status_array == "Arrivato")
    fuori_toll = np.sum(status_array == "Fuori Tolleranza")
    schiantati = np.sum(status_array == "Impatto")
    errori = np.sum(status_array == "Errore Solver")

    print("\n--- RESOCONTO STATISTICO DEFINITIVO ---")
    print(f"Successi Perfetti (Salvi e a target): {arrivati} ({(arrivati/NUM_TESTS)*100:.1f}%)")
    print(f"Fuori Tolleranza (Lenti ma salvi)   : {fuori_toll} ({(fuori_toll/NUM_TESTS)*100:.1f}%)")
    print(f"Impattati (Alpha > Limite stanza)  : {schiantati} ({(schiantati/NUM_TESTS)*100:.1f}%)")
    print(f"Errori (Crash Matematico Acados)    : {errori} ({(errori/NUM_TESTS)*100:.1f}%)")

    # ==========================================
    # GENERAZIONE PLOT CAMPIONE (10 TEST)
    # ==========================================
    voli_successo = [v for v in voli_completati if v["status"] == "Arrivato"]
    
    if len(voli_successo) > 0:
        num_campioni = min(10, len(voli_successo))
        campione_test = random.sample(voli_successo, num_campioni)
        
        cartella_plots = os.path.join("plots", "mpc")
        os.makedirs(cartella_plots, exist_ok=True)
        print(f"\nSalvataggio {num_campioni} plot in: '{cartella_plots}'")

        time_x = np.arange(len(campione_test[0]['x_hist'])) * DT
        time_u = np.arange(len(campione_test[0]['u_hist'])) * DT
        colors = plt.cm.tab10(np.linspace(0, 1, num_campioni))

        # 1. Posizioni e Angolo
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # 2. Velocità
        fig_vel, axs_vel = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # 3. Motori
        fig_mot, axs_mot = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # 4. Alpha
        fig_alpha, ax_alpha = plt.subplots(figsize=(10, 6))

        # --- SETUP GRAFICI ---
        # Posizioni
        axs_pos[0].set_title("Position x")
        axs_pos[0].set_ylabel("x [m]")
        axs_pos[0].set_xlabel("t [s]")

        axs_pos[1].set_title("Position z")
        axs_pos[1].set_ylabel("z [m]")
        axs_pos[1].set_xlabel("t [s]")

        axs_pos[2].set_title("Angle theta")
        axs_pos[2].set_ylabel("theta [rad]")
        axs_pos[2].set_xlabel("t [s]")


        # Velocità
        axs_vel[0].set_title("Velocity vx")
        axs_vel[0].set_ylabel("vx [m/s]")
        axs_vel[0].set_xlabel("t [s]")

        axs_vel[1].set_title("Velocity vz")
        axs_vel[1].set_ylabel("vz [m/s]")
        axs_vel[1].set_xlabel("t [s]")

        axs_vel[2].set_title("Velocity wy")
        axs_vel[2].set_ylabel("wy [rad/s]")
        axs_vel[2].set_xlabel("t [s]")


        # Motori
        axs_mot[0].set_title("Motor 1")
        axs_mot[0].set_ylabel("U1")
        axs_mot[0].set_xlabel("t [s]")

        axs_mot[1].set_title("Motor 2")
        axs_mot[1].set_ylabel("U2")
        axs_mot[1].set_xlabel("t [s]")

        # Alpha
        ax_alpha.set_title("Scaling factor Alpha")
        ax_alpha.set_ylabel("Alpha Predicted")
        ax_alpha.set_xlabel("t [s]")


        for res, color in zip(campione_test, colors):
            x_h = res['x_hist']
            u_h = res['u_hist']
            a_h = res['alpha_hist']
            xref = res['xref']
            lbl = f"Test #{res['id']}"

            # Posizioni
            axs_pos[0].plot(time_x, x_h[:, 0], color=color, label=lbl); axs_pos[0].axhline(xref[0], color=color, linestyle='--', alpha=0.3)
            axs_pos[1].plot(time_x, x_h[:, 1], color=color); axs_pos[1].axhline(xref[1], color=color, linestyle='--', alpha=0.3)
            axs_pos[2].plot(time_x, x_h[:, 2], color=color)
            
            # Velocità
            axs_vel[0].plot(time_x, x_h[:, 3], color=color, label=lbl)
            axs_vel[1].plot(time_x, x_h[:, 4], color=color)
            axs_vel[2].plot(time_x, x_h[:, 5], color=color)
            
            # Motori
            axs_mot[0].plot(time_u, u_h[:, 0], color=color, label=lbl)
            axs_mot[1].plot(time_u, u_h[:, 1], color=color)
            
            # Alpha
            ax_alpha.plot(time_u, a_h, color=color, label=lbl)

        # Setup Label e Salvataggio
        ax_alpha.axhline(ALPHA_REAL, color='red', linestyle='--', label='Alpha Real')
        fig_pos.savefig(os.path.join(cartella_plots, "1_posizioni.png"), dpi=300)
        fig_vel.savefig(os.path.join(cartella_plots, "2_velocita.png"), dpi=300)
        fig_mot.savefig(os.path.join(cartella_plots, "3_motori.png"), dpi=300)
        fig_alpha.savefig(os.path.join(cartella_plots, "4_alpha.png"), dpi=300)
        
        plt.close('all')
        print("Salvataggio completato!")

    # --- SCATTER PLOT FINALE ---
    plt.figure(figsize=(9, 6))
    valid_idx = ~np.isnan(err_array)
    
    # Assegnazione intelligente dei colori basata sullo status reale
    colori = []
    for stat in status_array[valid_idx]:
        if stat == "Arrivato":
            colori.append('blue')    # Salvo e a target
        elif stat == "Impatto":
            colori.append('red')     # Ha bucato la linea di Alpha
        else:
            colori.append('orange')  # Fuori tolleranza
            
    plt.scatter(err_array[valid_idx], alpha_array[valid_idx], alpha=0.6, c=colori, edgecolor='k')
    plt.axvline(TOLLERANZA_TARGET, color='green', linestyle='--', label=f'Target ({TOLLERANZA_TARGET}m)')
    plt.axhline(ALPHA_REAL, color='red', linestyle='--', linewidth=2, label=f'Limit ({ALPHA_REAL}m)')
    
    plt.title("Results Tests: Safety (Alpha) vs Accuracy (Error)")
    plt.xlabel("Position error from the target [m]")
    plt.ylabel("Max Alpha Request [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join("plots", "mpc", "0_scatter_statistico.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
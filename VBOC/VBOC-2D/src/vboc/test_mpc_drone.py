import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork

# Importa le tue librerie (adatta i percorsi se necessario)
from parser import Parameters, parse_args  # Assumendo che il parser sia accessibile
from mpc_abstract import Model
from mpc_controller import MpcController

def main():
    # --- 1. SETUP PARAMETRI E CONTROLLER ---
    print("--- Inizializzazione Sistema ---")
    # Puoi forzare il nome del robot e l'attivazione se non usi il terminale
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' # Assicurati che corrisponda al tuo training!
    params.build = True # Compila il solver C alla prima esecuzione

    # Inizializziamo Modello e Controller MPC
    model = Model(params)
    controller = MpcController(model)

    # --- 2. CONFIGURAZIONE SIMULAZIONE ---
    DT = params.dt
    SIM_TIME = 5.0  # Secondi totali di volo
    N_SIM = int(SIM_TIME / DT)

    # Stato iniziale del drone [x, z, theta, vx, vz, wy]
    # Partiamo dall'origine, fermi.
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Target da raggiungere [x, z, theta, vx, vz, wy]
    # Vogliamo andare a x=3 metri, z=2 metri, e fermarci lì (velocità 0)
    x_ref = np.array([3.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    # Lettura simulata dei sensori: quanti metri di spazio abbiamo intorno?
    # Fissiamo la stanza a 1.5 metri per ora.
    alpha_real_sensor = 1.5 

    print(f"\nStato Iniziale: {x0}")
    print(f"Target: {x_ref}")
    print(f"Spazio di sicurezza letto dai sensori (Alpha Real): {alpha_real_sensor}")

    # Variabili per salvare la storia del volo
    x_history = [x0]
    u_history = []
    alpha_history = []
    
    current_x = x0.copy()

    # --- 3. IL CICLO DI VOLO (MPC LOOP) ---
    print("\nAvvio loop di controllo MPC...")
    start_time = time.time()

    # --- WARM START ---
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    for t in range(N_SIM):
        # Chiama il nostro bellissimo solve_step!
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, alpha_real_sensor)
        
        if status not in [0, 2]:
            print(f"Errore Solver al passo {t}! Status: {status}. Il drone non sa cosa fare.")
            break
      
            
        # Estraiamo il primo comando ottimale calcolato per i motori
        u_cmd = u_sol[0]
        
        # --- APPLICAZIONE DELLA FISICA ---
        # In una simulazione reale perfetta senza disturbi, il passo successivo (x_next)
        # è esattamente quello che il solver ha previsto al nodo 1.
        # Altrimenti, qui andrebbe la funzione di integrazione RK4.
        next_x = x_sol[1] 
        
        # Salviamo i dati
        x_history.append(next_x)
        u_history.append(u_cmd)
        alpha_history.append(alpha_curr)
        
        
        # Aggiorniamo la posizione del drone per il prossimo ciclo
        current_x = next_x
        
        # Stampa di debug a OGNI SINGOLO STEP (così vedi che lavora!)
        print(f"Step {t:03d} | X={current_x[0]:.2f} Z={current_x[1]:.2f} | Alpha_Pred={alpha_curr:.3f}")


    end_time = time.time()
    print(f"\nSimulazione terminata in {end_time - start_time:.2f} secondi.")

    # --- 4. PLOT DEI RISULTATI ---
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    time_axis = np.arange(len(x_history)) * DT

    if len(u_history) == 0:
        print("Nessun dato da plottare.")
        return

    # Plot Posizioni X e Z e angle
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, x_history[:, 0], label='X Drone', color='b')
    plt.plot(time_axis, x_history[:, 1], label='Z Drone', color='g')
    plt.plot(time_axis, x_history[:, 2], label='theta Drone', color='r')
    plt.axhline(x_ref[0], color='b', linestyle='--', alpha=0.5, label='Target X')
    plt.axhline(x_ref[1], color='g', linestyle='--', alpha=0.5, label='Target Z')
    plt.axhline(x_ref[2], color='r', linestyle='--', alpha=0.5, label='Target theta')
    plt.title('Traiettoria (Inseguimento Target)')
    plt.xlabel('Time [s]')
    plt.ylabel('Pose [m]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Velocità Vx e Vz
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, x_history[:, 3], label='Vx (Orizzontale)', color='b')
    plt.plot(time_axis, x_history[:, 4], label='Vz (Verticale)', color='g')
    plt.plot(time_axis, x_history[:, 5], label='wy (Verticale)', color='r')
    # Linee di riferimento per il range della rete neurale (+/- 1 m/s)
    plt.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='Limite Rete (1.0 m/s)')
    plt.axhline(x_ref[3], color='b', linestyle='--', alpha=0.5, label='Target Vx')
    plt.axhline(x_ref[4], color='g', linestyle='--', alpha=0.5, label='Target Vz')
    plt.axhline(x_ref[5], color='r', linestyle='--', alpha=0.5, label='Target wy')
    plt.axhline(-1.0, color='k', linestyle=':', alpha=0.5)
    plt.title('Andamento Velocità (Range Rete ±1m/s)')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocità [m/s]')
    plt.legend()
    plt.grid(True)

    # Plot Input Motori
    plt.figure(figsize=(10, 5))
    valid_len = len(u_history)
    plt.plot(time_axis[:valid_len], u_history[:, 0], label='Motore 1')
    plt.plot(time_axis[:valid_len], u_history[:, 1], label='Motore 2')
    plt.axhline(model.u_bar, color='r', linestyle='--', alpha=0.5, label='Max Power')
    plt.title('Comando Motori (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Spinta [$u^2$]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot alpha
    plt.figure(figsize=(10, 5))
    # Creiamo un asse temporale per alpha (lungo quanto u_history)
    time_alpha = np.arange(len(alpha_history)) * DT
    plt.plot(time_alpha, alpha_history, label='Alpha Predetto (NN)', color='purple', linewidth=2)
    plt.axhline(alpha_real_sensor, color='red', linestyle='--', label=f'Alpha Reale ({alpha_real_sensor}m)')
    plt.title('Monitoraggio Sicurezza (Vincolo Terminale)')
    plt.ylabel('Distanza Alpha [m]')
    plt.xlabel('Tempo [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
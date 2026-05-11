import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork
import matplotlib.patches as patches


# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

# Importo librerie
from parser import Parameters, parse_args  #
from mpc_abstract import Model
from mpc_controller import MpcController


def main():
    # --- SETUP PARAMETRI E CONTROLLER ---
    print("--- Inizializzazione Sistema ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu'
    params.build = True

    # Inizializzo Modello e Controller MPC
    model = Model(params)
    controller = MpcController(model)

    # --- CONFIGURAZIONE SIMULAZIONE ---
    DT = params.dt
    SIM_TIME = 5.5  # Secondi totali di volo
    N_SIM = int(SIM_TIME / DT)

    # Stato iniziale del drone [x, z, theta, vx, vz, wy]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Target da raggiungere [x, z, theta, vx, vz, wy]
    x_ref = np.array([15.7, 0.0, 0.0, 0.0, 0.0, 0.0])


    # Lettura simulata dei sensori
    # Fisso la stanza per ora
    X_muro_fisso = 16.0

    print(f"\nStato Iniziale: {x0}")
    print(f"Target: {x_ref}")
    #print(f"Spazio di sicurezza letto dai sensori (Alpha Real): {alpha_real_sensor}")

    # Variabili per salvare la storia del volo
    x_history = [x0]
    u_history = []
    alpha_history = []
    
    current_x = x0.copy()

    # --- MPC LOOP ---
    print("\nAvvio loop di controllo MPC...")
    start_time = time.time()

    # --- WARM START ---
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    for t in range(N_SIM):
        

        # 3. Passo la posizione al solver
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, X_muro_fisso)

        # Controllo fisico di schianto
        if current_x[0] >= X_muro_fisso:
            print(f"💥 CRASH! Il drone ha colpito il muro al passo {t}! Posizione: {current_x[0]:.2f}")
            break
        
        if status not in [0, 2]:
            print(f"Errore Solver al passo {t}! Status: {status}. Il drone non sa cosa fare.")
            break
      
        u_cmd = u_sol[0]
        next_x = x_sol[1] 
        
        # Salvataggio dati
        x_history.append(next_x)
        u_history.append(u_cmd)
        alpha_history.append(alpha_curr)
        
        
        # Aggiorno la posizione del drone per il prossimo ciclo
        current_x = next_x
        
        # Stampa di debug a ogni step
        print(f"Step {t:03d} | X={current_x[0]:.2f} Z={current_x[1]:.2f} | Alpha_Pred={alpha_curr:.3f}")


    end_time = time.time()
    print(f"\nSimulazione terminata in {end_time - start_time:.2f} secondi.")

    # --- PLOT RISULTATI ---
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    time_axis = np.arange(len(x_history)) * DT

    if len(u_history) == 0:
        print("Nessun dato da plottare.")
        return

    # Plot Posizioni
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, x_history[:, 0], label='X Drone', color='b')
    plt.plot(time_axis, x_history[:, 1], label='Z Drone', color='g')
    plt.plot(time_axis, x_history[:, 2], label='Theta Drone', color='r')
    plt.axhline(x_ref[0], color='b', linestyle='--', alpha=0.5, label='Target X')
    plt.axhline(x_ref[1], color='g', linestyle='--', alpha=0.5, label='Target Z')
    plt.axhline(x_ref[2], color='r', linestyle='--', alpha=0.5, label='Target theta')
    plt.title('Trajectory (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Pose [m]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plot Velocità
    plt.figure(figsize=(11, 6))
    plt.plot(time_axis, x_history[:, 3], label='Vx', color='b')
    plt.plot(time_axis, x_history[:, 4], label='Vz', color='g')
    plt.plot(time_axis, x_history[:, 5], label='wy', color='r')
    plt.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='NN Limit')
    plt.axhline(x_ref[3], color='b', linestyle='--', alpha=0.5, label='Target Vx')
    plt.axhline(x_ref[4], color='g', linestyle='--', alpha=0.5, label='Target Vz')
    plt.axhline(x_ref[5], color='r', linestyle='--', alpha=0.5, label='Target wy')
    plt.axhline(-1.0, color='k', linestyle=':', alpha=0.5)
    plt.title('Velocity (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot Input Motori
    plt.figure(figsize=(10, 6))
    valid_len = len(u_history)
    plt.plot(time_axis[:valid_len], u_history[:, 0], label='Motor 1')
    plt.plot(time_axis[:valid_len], u_history[:, 1], label='Motor 2')
    plt.axhline(model.u_bar, color='r', linestyle='--', alpha=0.5, label='Max Power')
    plt.title('Motors (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('$u^2$ [(Hz/s)$^2$]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plot alpha
    # Plot Security Monitoring
    plt.figure(figsize=(11, 6))
    time_alpha = np.arange(len(alpha_history)) * DT
    
    # Calcoliamo la distanza EFFETTIVA dal muro istante per istante
    distanza_reale = X_muro_fisso - x_history[:-1, 0] # [:-1] per pareggiare gli array
    
    plt.plot(time_alpha, alpha_history, label='Predicted space requested', color='purple', linewidth=2)
    plt.plot(time_alpha, distanza_reale, label='Real distance to the wall', color='red', linestyle='--')
    
    plt.title('Security Monitoring: Requested vs Available Space')
    plt.ylabel('Distance [m]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- NUOVO PLOT: VISUALIZZAZIONE SPAZIALE ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 1. Disegniamo il Muro come un blocco grigio
    # Assumiamo un corridoio alto da -1 a 1 metro per visualizzazione
    muro = patches.Rectangle((X_muro_fisso, -1.0), 0.5, 2, facecolor='gray', alpha=0.7, label='Obstacle')
    ax.add_patch(muro)

    # # 2. Disegniamo il corridoio (linee tratteggiate per i limiti Z)
    # plt.axhline(1, color='black', linestyle='-', alpha=0.3)
    # plt.axhline(-1, color='black', linestyle='-', alpha=0.3)

    # 3. Traiettoria del drone
    plt.plot(x_history[:, 0], x_history[:, 1], color='blue', linewidth=2, label='Drone Trajectory', marker='o', markersize=3)

    # 4. Start e Target
    plt.scatter(x0[0], x0[1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_ref[0], x_ref[1], color='red', s=100, label='Target', marker='X', zorder=5)

    # Configurazione assi
    plt.title('Drone Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    # plt.xlim([-0.5, X_muro_fisso + 1])
    # plt.ylim([-1.5, 1.5])
    #plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Importo librerie
from parser import Parameters, parse_args  #
from mpc_abstract import Model
from mpc_controller import MpcController

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

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

    # Stato iniziale del drone in 3D: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Target da raggiungere [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    x_ref = np.array([15.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    u_hover = (model.mass * 9.81) / (4.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    for t in range(N_SIM):

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
        print(f"Step {t:03d} | X={current_x[0]:.2f} Y={current_x[1]:.2f} Z={current_x[2]:.2f} | Alpha_Pred={alpha_curr:.3f}")


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
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, x_history[:, 0], label='X Drone', color='b')
    plt.plot(time_axis, x_history[:, 1], label='Y Drone', color='orange')
    plt.plot(time_axis, x_history[:, 2], label='Z Drone', color='g')
    plt.axhline(x_ref[0], color='b', linestyle='--', alpha=0.5, label='Target X')
    plt.axhline(x_ref[1], color='orange', linestyle='--', alpha=0.5, label='Target Y')
    plt.axhline(x_ref[2], color='g', linestyle='--', alpha=0.5, label='Target Z')
    plt.title('Trajectory (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Angoli
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, np.rad2deg(x_history[:, 3]), label='Roll (phi)', color='r')
    plt.plot(time_axis, np.rad2deg(x_history[:, 4]), label='Pitch (theta)', color='g')
    plt.plot(time_axis, np.rad2deg(x_history[:, 5]), label='Yaw (psi)', color='b')
    plt.title('Euler Angles (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Velocità Lineari
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, x_history[:, 6], label='Vx', color='b')
    plt.plot(time_axis, x_history[:, 7], label='Vy', color='orange')
    plt.plot(time_axis, x_history[:, 8], label='Vz', color='g')
    plt.axhline(x_ref[6], color='k', linestyle='--', alpha=0.5, label='Target Vel')
    plt.title('Linear Velocities (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Input Motori
    plt.figure(figsize=(12, 5))
    valid_len = len(u_history)
    plt.plot(time_axis[:valid_len], u_history[:, 0], label='Motor 1 (FR)')
    plt.plot(time_axis[:valid_len], u_history[:, 1], label='Motor 2 (FL)')
    plt.plot(time_axis[:valid_len], u_history[:, 2], label='Motor 3 (RL)')
    plt.plot(time_axis[:valid_len], u_history[:, 3], label='Motor 4 (RR)')
    plt.axhline(model.u_bar, color='r', linestyle='--', alpha=0.5, label='Max Power')
    plt.title('Motors (MPC)')
    plt.xlabel('Time [s]')
    plt.ylabel('$u^2$ [(Hz/s)$^2$]')
    plt.legend()
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

    # --- NUOVO PLOT: VISUALIZZAZIONE SPAZIALE 3D ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Disegniamo il Muro come un blocco grigio 3D
    # Definiamo i limiti dell'ostacolo. 
    # Assumiamo che il muro vada da Y=-2 a Y=2 e in altezza da Z=-1 a Z=1
    x_min = X_muro_fisso
    x_max = X_muro_fisso + 0.5
    y_min = -2.0  
    y_max = 2.0
    z_min = -1.0  
    z_max = 1.0

    # Creiamo gli 8 vertici del parallelepipedo
    v = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])

    # Definiamo le 6 facce collegando i vertici
    faces = [
        [v[0], v[1], v[2], v[3]], # Base
        [v[4], v[5], v[6], v[7]], # Coperchio
        [v[0], v[1], v[5], v[4]], # Faccia frontale
        [v[2], v[3], v[7], v[6]], # Faccia posteriore
        [v[1], v[2], v[6], v[5]], # Faccia destra
        [v[4], v[7], v[3], v[0]]  # Faccia sinistra
    ]

    # Aggiungiamo il muro 3D al plot
    muro_3d = Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='black', alpha=0.5, label='Obstacle')
    ax.add_collection3d(muro_3d)

    # (Fix per la legenda di Matplotlib con oggetti 3D)
    muro_3d._facecolors2d = muro_3d._facecolor3d
    muro_3d._edgecolors2d = muro_3d._edgecolor3d

    # 2. Traiettoria del drone in 3D (X, Y, Z)
    ax.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], color='blue', linewidth=2, label='Drone Trajectory', marker='o', markersize=3)

    # 3. Start e Target
    ax.scatter(x0[0], x0[1], x0[2], color='green', s=100, label='Start')
    ax.scatter(x_ref[0], x_ref[1], x_ref[2], color='red', s=100, marker='X', label='Target')

    # Impostiamo le etichette e le proporzioni degli assi
    ax.set_xlim([-1, X_muro_fisso + 2])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Visualizzazione Spaziale 3D della Traiettoria')

    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
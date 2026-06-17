import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork
import matplotlib.patches as patches


# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for everything else
})

# Import libraries
from parser import Parameters, parse_args  #
from mpc_abstract import Model
from mpc_controller import MpcController


def main():
    # --- PARAMETER AND CONTROLLER SETUP ---
    print("--- Initializing system ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu'
    params.build = True

    # Inizializzo Modello e Controller MPC
    model = Model(params)
    controller = MpcController(model)

    # --- SIMULATION CONFIGURATION ---
    DT = params.dt
    SIM_TIME = 5.5  # Total flight time in seconds
    N_SIM = int(SIM_TIME / DT)

    # Initial drone state [x, z, theta, vx, vz, wy]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Reference target [x, z, theta, vx, vz, wy]
    x_ref = np.array([15.7, 0.0, 0.0, 0.0, 0.0, 0.0])


    # Simulated sensor reading
    # Fix the wall for now
    X_muro_fisso = 16.0

    print(f"\nStato Iniziale: {x0}")
    print(f"Target: {x_ref}")
    #print(f"Spazio di sicurezza letto dai sensori (Alpha Real): {alpha_real_sensor}")

    # Variables to save flight history
    x_history = [x0]
    u_history = []
    alpha_history = []
    
    current_x = x0.copy()

    # --- MPC LOOP ---
    print("\nStarting MPC control loop...")
    start_time = time.time()

    # --- WARM START ---
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    for t in range(N_SIM):
        

        # 3. Pass the current position to the solver
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, X_muro_fisso)

        # Physical crash check
        if current_x[0] >= X_muro_fisso:
            print(f"💥 CRASH! Il drone ha colpito il muro al passo {t}! Posizione: {current_x[0]:.2f}")
            break
        
        if status not in [0, 2]:
            print(f"Errore Solver al passo {t}! Status: {status}. Il drone non sa cosa fare.")
            break
      
        u_cmd = u_sol[0]
        next_x = x_sol[1] 
        
        # Save data
        x_history.append(next_x)
        u_history.append(u_cmd)
        alpha_history.append(alpha_curr)
        
        
        # Update drone position for the next cycle
        current_x = next_x
        
        # Debug print each step
        print(f"Step {t:03d} | X={current_x[0]:.2f} Z={current_x[1]:.2f} | Alpha_Pred={alpha_curr:.3f}")


    end_time = time.time()
    print(f"\nSimulazione terminata in {end_time - start_time:.2f} secondi.")

    # --- PLOT RESULTS ---
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    time_axis = np.arange(len(x_history)) * DT

    if len(u_history) == 0:
        print("Nessun dato da plottare.")
        return

    # Plot positions
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

    # Plot velocities
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

    # Plot motor inputs
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
    # Plot security monitoring
    plt.figure(figsize=(11, 6))
    time_alpha = np.arange(len(alpha_history)) * DT
    
    # Compute the actual distance to the wall at each timestep
    distanza_reale = X_muro_fisso - x_history[:-1, 0] # [:-1] to align the arrays
    
    plt.plot(time_alpha, alpha_history, label='Predicted space requested', color='purple', linewidth=2)
    plt.plot(time_alpha, distanza_reale, label='Real distance to the wall', color='red', linestyle='--')
    
    plt.title('Security Monitoring: Requested vs Available Space')
    plt.ylabel('Distance [m]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- NEW PLOT: SPATIAL VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 1. Draw the wall as a gray block
    # Assume a corridor height from -1 to 1 meter for visualization
    muro = patches.Rectangle((X_muro_fisso, -1.0), 0.5, 2, facecolor='gray', alpha=0.7, label='Obstacle')
    ax.add_patch(muro)

    # # 2. Draw the corridor (dashed lines for the Z limits)
    # plt.axhline(1, color='black', linestyle='-', alpha=0.3)
    # plt.axhline(-1, color='black', linestyle='-', alpha=0.3)

    # 3. Drone trajectory
    plt.plot(x_history[:, 0], x_history[:, 1], color='blue', linewidth=2, label='Drone Trajectory', marker='o', markersize=3)

    # 4. Start and Target
    plt.scatter(x0[0], x0[1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_ref[0], x_ref[1], color='red', s=100, label='Target', marker='X', zorder=5)

    # Axis configuration
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
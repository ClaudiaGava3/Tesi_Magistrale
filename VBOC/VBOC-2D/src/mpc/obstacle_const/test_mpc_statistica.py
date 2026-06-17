import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Import your libraries
from parser import Parameters
from mpc_abstract import Model
from mpc_controller import MpcController

# always 400 tests

# short horizon N=15

# speed between +-1 and angles between +-pi/4
# obstacle at 4m: NN->95.5, naive->95.8
# obstacle at 6m: NN->87.2, naive->87.5
# obstacle at 8m: NN->85, naive->34
# obstacle at 10m: NN->82.2, naive->1.8
# obstacle at 12m: NN->79.2, naive->0.2

# zero speed and angles
# obstacle at 4m: NN->100, naive->100
# obstacle at 6m: NN->100, naive->93.8
# obstacle at 8m: NN->100, naive->62.0
# obstacle at 10m: NN->100, naive->0.0
# obstacle at 12m: NN->100, naive->3.2


# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for everything else
})

def genera_condizioni_iniziali(x_muro_fisso):
    """
    Generate random initial states and target positions
    following exactly the definition from Chapter 5.2.2.
    """
    # 1. Fixed initial position (origin)
    x0_pos = 0.0
    z0_pos = 0.0 
    
    # 2. Random angles and velocities
    theta0 = np.random.uniform(-np.pi/4, np.pi/4) 
    vx0 = np.random.uniform(-1.0, 1.0) 
    vz0 = np.random.uniform(-1.0, 1.0)
    wy0 = np.random.uniform(-1.0, 1.0)
    
    x0 = np.array([x0_pos, z0_pos, theta0, vx0, vz0, wy0])
    
    # 3. Random obstacle delta between 0.2 and 0.6 meters
    delta_obs = np.random.uniform(0.2, 0.6)
    
    # 4. Target position: just before the obstacle
    xref_pos = x_muro_fisso - delta_obs
    zref_pos = 0.0  # Volo in linea retta
    
    x_ref = np.array([xref_pos, zref_pos, 0.0, 0.0, 0.0, 0.0])
    
    return x0, x_ref, delta_obs

def main():
    print("--- Inizializzazione Campagna Test ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' 
    params.build = True

    model = Model(params)
    controller = MpcController(model)

    # --- SIMULATION PARAMETERS ---
    NUM_TESTS = 400
    DT = params.dt
    SIM_TIME = 3.5
    N_SIM = int(SIM_TIME / DT)
    
    TOLLERANZA_TARGET = 0.1  # Errore massimo accettabile dal target
    X_MURO_FISSO = 8.0       # Ostacolo sufficientemente lontano

    # Statistical counters
    arrivati = 0
    impattati = 0
    falliti_matematicamente = 0

    # Lists to save successful flights (for plotting)
    voli_successo = []

    print(f"\nAvvio {NUM_TESTS} simulazioni. Volo: {SIM_TIME}s, Ostacolo a: {X_MURO_FISSO}m")
    start_total = time.time()

    for i in range(NUM_TESTS):
        x0, x_ref, delta_obs = genera_condizioni_iniziali(X_MURO_FISSO)
        current_x = x0.copy()
        
        solver_fallito = False
        impatto_fisico = False
        
        # Local history for potential plotting
        x_hist = [current_x]
        
        # Warm start
        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        x_guess_safe = np.array([current_x[0], current_x[1], 0.0, 0.0, 0.0, 0.0])
        
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x_guess_safe, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # Simulation loop for the single flight
        for t in range(N_SIM):
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, X_MURO_FISSO)
            
            # Real physical impact check
            if current_x[0] >= X_MURO_FISSO:
                impatto_fisico = True
                break
                
            # Mathematical failure check of the MPC (Status 4)
            if status not in [0, 2]:
                solver_fallito = True
                break
                
            current_x = x_sol[1]
            x_hist.append(current_x)

        # --- FINAL FLIGHT EVALUATION ---
        # If it impacted or crossed the wall at the last moment
        if impatto_fisico or current_x[0] >= X_MURO_FISSO:
            impattati += 1
            esito = "Impatto"
        # If the optimizer panicked (Naive myopia)
        elif solver_fallito:
            falliti_matematicamente += 1
            esito = "Infeasible (Status 4)"
        # If it survived, check whether it reached the target
        else:
            errore_pos = np.abs(current_x[0] - x_ref[0])
            if errore_pos <= TOLLERANZA_TARGET:
                arrivati += 1
                esito = "Successo"
                
                # Save data for plotting (up to 10)
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

        # Progress bar
        percentuale = (i + 1) / NUM_TESTS
        barra = '█' * int(30 * percentuale) + '-' * (30 - int(30 * percentuale))
        sys.stdout.write(f"\rProgresso: [{barra}] {percentuale*100:.1f}% ({i+1}/{NUM_TESTS})")
        sys.stdout.flush()

    end_total = time.time()
    print(f"\n\nCampagna completata in {(end_total - start_total)/60:.1f} minuti.")

    # --- STATISTICAL SUMMARY ---
    print("\n================================================")
    print("           RESOCONTO STATISTICO FINALE          ")
    print("================================================")
    print(f"Test Totali             : {NUM_TESTS}")
    print(f"Successi (A Target)     : {arrivati} ({(arrivati/NUM_TESTS)*100:.1f}%)")
    print(f"Impatti Fisici contro ostacolo : {impattati} ({(impattati/NUM_TESTS)*100:.1f}%)")
    print(f"Fallimenti (Infeasible/Status4): {falliti_matematicamente} ({(falliti_matematicamente/NUM_TESTS)*100:.1f}%)")
    print("================================================\n")

    # --- GENERATE PLOTS FOR THE 10 SUCCESSFUL TESTS ---
    if len(voli_successo) > 0:
        cartella_plots = os.path.join("plots", "statistica")
        os.makedirs(cartella_plots, exist_ok=True)
        print(f"Salvataggio plot di {len(voli_successo)} voli di successo in: '{cartella_plots}'")

        plt.figure(figsize=(12, 6))
        
        # Draw the fixed wall on the right
        plt.axvspan(X_MURO_FISSO, X_MURO_FISSO + 1.0, color='gray', alpha=0.5, label='Ostacolo')

        colors = plt.cm.tab10(np.linspace(0, 1, len(voli_successo)))

        for res, color in zip(voli_successo, colors):
            x_h = res['x_hist']
            xref = res['xref']
            
            # Trajectory in the X-Z plane
            plt.plot(x_h[:, 0], x_h[:, 1], color=color, linewidth=2, alpha=0.8)
            # Mark the target for this specific trajectory
            plt.scatter(xref[0], xref[1], color=color, marker='X', s=100, zorder=5)

        plt.title(f'Trajectory of {len(voli_successo)} test with Success')
        plt.xlabel('Position X [m]')
        plt.ylabel('Position Z [m]')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Generic legend labels
        plt.plot([], [], color='black', linewidth=2, label='Trajectory Drone')
        plt.scatter([], [], color='black', marker='X', s=100, label='Target')
        #plt.legend(loc='lower left')

        plt.savefig(os.path.join(cartella_plots, "10_test_successo.png"), dpi=300)
        plt.show()
    else:
        print("Nessun volo di successo da plottare. (Questo è normale se stai testando il Naive MPC!)")

if __name__ == '__main__':
    main()
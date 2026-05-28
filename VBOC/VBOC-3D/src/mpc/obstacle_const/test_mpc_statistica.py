import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import random
from learning import NeuralNetwork
from mpl_toolkits.mplot3d.art3d import Poly3DCollection #
from matplotlib.lines import Line2D

# Importa le tue librerie
from parser import Parameters
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


def genera_condizioni_iniziali(x_muro_fisso):
    """Genera stati fisicamente possibili per non far impazzire il solver."""  
    
    # Partenza da (0,0,0) ma con angoli e velocità randomici in 3D
    x0_pos, y0_pos, z0_pos = 0.0, 0.0, 0.0

    # Angoli Roll, Pitch, Yaw
    phi0 = np.random.uniform(-np.radians(45), np.radians(45))
    theta0 = np.random.uniform(-np.radians(45), np.radians(45))
    psi0 = np.random.uniform(-np.radians(45), np.radians(45))

    # Velocità e tassi angolari
    vx0, vy0, vz0 = np.random.uniform(-1.0, 1.0, 3) 
    p0, q0, r0 = np.random.uniform(-1.0, 1.0, 3)
        
          
    delta_obs = np.random.uniform(0.2, 0.6)

    xref_pos = x_muro_fisso - delta_obs
    yref_pos = 0.0
    zref_pos = 0.0  # Volo in linea rett
            
    x0 = np.array([x0_pos, y0_pos, z0_pos, phi0, theta0, psi0, vx0, vy0, vz0, p0, q0, r0])
    x_ref = np.array([xref_pos, yref_pos, zref_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    
    return x0, x_ref, delta_obs

def main():
    print("--- Inizializzazione Campagna Test ---")
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
        
        # Storia locale per il singolo volo
        x_hist = [current_x]
        
        # Warm start
        u_hover = (model.mass * 9.81) / (4.0 * model.cf)
        x_guess_safe = np.zeros(12)
        x_guess_safe[:3] = x0[:3]
        
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x_guess_safe, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # Loop di simulazione
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

    # ==========================================
    # GENERAZIONE PLOT CAMPIONE (10 TEST)
    # ==========================================
        
    if len(voli_successo) > 0:
        cartella_plots = os.path.join("plots", "statistica")
        os.makedirs(cartella_plots, exist_ok=True)
        print(f"Salvataggio plot di {len(voli_successo)} voli di successo in: '{cartella_plots}'")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d') #
        
        # 1. Disegno del muro fisso in 3D
        # Ipotizziamo dei limiti per Y e Z per mostrare il muro come una lastra verticale
        x_m_min, x_m_max = X_MURO_FISSO, X_MURO_FISSO + 0.5
        y_m_min, y_m_max = -4.0, 4.0  # Copre l'area di volo su Y
        z_m_min, z_m_max = -4.0, 4.0  # Copre l'area di volo su Z
        
        # Definiamo gli 8 vertici del muro
        v = np.array([[x_m_min, y_m_min, z_m_min], [x_m_max, y_m_min, z_m_min], [x_m_max, y_m_max, z_m_min], [x_m_min, y_m_max, z_m_min],
                      [x_m_min, y_m_min, z_m_max], [x_m_max, y_m_min, z_m_max], [x_m_max, y_m_max, z_m_max], [x_m_min, y_m_max, z_m_max]])
        faces = [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]], [v[0],v[1],v[5],v[4]], 
                 [v[2],v[3],v[7],v[6]], [v[1],v[2],v[6],v[5]], [v[4],v[7],v[3],v[0]]]
        
        muro_3d = Poly3DCollection(faces, facecolors='gray', alpha=0.3, edgecolors='black')
        ax.add_collection3d(muro_3d)

        colors = plt.cm.tab10(np.linspace(0, 1, len(voli_successo)))

        for res, color in zip(voli_successo, colors):
            x_h = res['x_hist']
            xref = res['xref']
            
            # Traiettoria 3D (indici 0, 1, 2 per X, Y, Z)
            ax.plot(x_h[:, 0], x_h[:, 1], x_h[:, 2], color=color, linewidth=2, alpha=0.8)
            # Segnamo il target in 3D
            ax.scatter(xref[0], xref[1], xref[2], color=color, marker='X', s=100, zorder=5)

        ax.set_title(f'Trajectory of {len(voli_successo)} 3D tests with Success')
        ax.set_xlabel('Position X [m]')
        ax.set_ylabel('Position Y [m]')
        ax.set_zlabel('Position Z [m]')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Creazione manuale della legenda per gli oggetti 3D
        legend_elements = [Line2D([0], [0], color='black', lw=2, label='Trajectory Drone'),
                           Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='Target')]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.savefig(os.path.join(cartella_plots, "10_test_successo_3D.png"), dpi=300)
        plt.show()

if __name__ == '__main__':
    main()
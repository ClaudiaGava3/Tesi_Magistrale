import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from learning import NeuralNetwork

# Importa le tue librerie (adatta i percorsi se necessario)
from parser import Parameters
from mpc_abstract import Model
from mpc_controller import MpcController

def esegui_volo(controller, params, x0, x_ref, alpha_real_sensor):
    """Esegue una singola simulazione di volo e restituisce le storie."""
    DT = params.dt
    SIM_TIME = 5.0  # Secondi totali di volo
    N_SIM = int(SIM_TIME / DT)

    x_history = [x0]
    u_history = []
    alpha_history = []
    
    current_x = x0.copy()

    # --- WARM START E RESET PER OGNI VOLO ---
    u_hover = (controller.model.mass * 9.81) / (2.0 * controller.model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, controller.model.nu), u_hover)

    for t in range(N_SIM):
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, alpha_real_sensor)
        
        if status not in [0, 2]:
            print(f"    [!] Errore Solver al passo {t}! Status: {status}")
            return None, None, None, False # Fallimento
            
        u_cmd = u_sol[0]
        next_x = x_sol[1] 
        
        x_history.append(next_x)
        u_history.append(u_cmd)
        alpha_history.append(alpha_curr)
        current_x = next_x

    return np.array(x_history), np.array(u_history), np.array(alpha_history), True

def main():
    print("--- Inizializzazione Sistema (Stress Test) ---")
    robot_name = 'sth'
    params = Parameters(robot_name)
    params.act = 'gelu' 
    params.build = True 

    model = Model(params)
    controller = MpcController(model)

    # ==========================================
    # DEFINIZIONE DEI 10 SCENARI DI TEST
    # ==========================================
    # Formato: [x, z, theta, vx, vz, wy]
    test_cases = [
        {"nome": "1. Standard",      "x0": [0,0,0,0,0,0],  "xref": [3, 2,0,0,0,0]},
        {"nome": "2. Salita pura",   "x0": [0,0,0,0,0,0],  "xref": [0, 3,0,0,0,0]},
        {"nome": "3. Discesa in fwd","x0": [0,3,0,0,0,0],  "xref": [3, 0,0,0,0,0]},
        {"nome": "4. Volo indietro", "x0": [3,2,0,0,0,0],  "xref": [0, 0,0,0,0,0]},
        {"nome": "5. Diagonale sx",  "x0": [0,0,0,0,0,0],  "xref": [-3,2,0,0,0,0]},
        {"nome": "6. Hovering fisso","x0": [1,1,0,0,0,0],  "xref": [1, 1,0,0,0,0]},
        {"nome": "7. Rasoterra",     "x0": [0,0.5,0,0,0,0],"xref": [4, 0.5,0,0,0,0]},
        {"nome": "8. Scatto breve",  "x0": [0,0,0,0,0,0],  "xref": [1, 1,0,0,0,0]},
        {"nome": "9. Discesa pura",  "x0": [0,3,0,0,0,0],  "xref": [0, 1,0,0,0,0]},
        {"nome": "10. Manovra a V",  "x0": [-2,3,0,0,0,0], "xref": [2, 3,0,0,0,0]}
    ]

    alpha_fisso = 1.5 
    risultati = []

    print("\n--- Inizio Esecuzione Test Multipli ---")
    
    for i, caso in enumerate(test_cases):
        print(f"\nEseguo Test {caso['nome']} | Da {caso['x0']} a {caso['xref']}")
        start_time = time.time()
        
        x_hist, u_hist, alpha_hist, success = esegui_volo(
            controller, params, 
            np.array(caso['x0'], dtype=float), 
            np.array(caso['xref'], dtype=float), 
            alpha_fisso
        )
        
        end_time = time.time()
        
        if success:
            distanza_finale = np.linalg.norm(x_hist[-1][:2] - caso['xref'][:2])
            print(f"    -> Successo! Tempo calcolo: {end_time-start_time:.2f}s | Errore target: {distanza_finale:.3f}m")
            risultati.append({
                "nome": caso['nome'],
                "x_hist": x_hist,
                "u_hist": u_hist,
                "alpha_hist": alpha_hist,
                "xref": caso['xref']
            })
        else:
            print("    -> FALLITO.")

    # ==========================================
    # PLOT MULTIPLI
    # ==========================================
    if not risultati:
        print("Nessun test da plottare.")
        return

    time_x = np.arange(len(risultati[0]['x_hist'])) * params.dt
    time_u = np.arange(len(risultati[0]['u_hist'])) * params.dt # u_hist ha 1 elemento in meno di x_hist
    colors = plt.cm.tab10(np.linspace(0, 1, len(risultati)))

    # --- FIGURA 1: Mappa X-Z e Sicurezza Alpha ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_title("Mappa Traiettorie (X-Z)")
    ax1.set_xlabel("Posizione X [m]"); ax1.set_ylabel("Posizione Z [m]"); ax1.grid(True)
    
    ax2.set_title("Monitoraggio Alpha (Spazio di sicurezza)")
    ax2.set_xlabel("Tempo [s]"); ax2.set_ylabel("Alpha Predetto [m]")
    ax2.axhline(alpha_fisso, color='red', linestyle='--', linewidth=2, label=f"Limite Stanza ({alpha_fisso}m)")
    ax2.grid(True)

    # --- FIGURA 2: Posizioni nel Tempo (X e Z separati) ---
    fig2, (ax_px, ax_pz) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle("Andamento Posizioni nel Tempo", fontsize=14)
    ax_px.set_ylabel("Posizione X [m]"); ax_px.grid(True)
    ax_pz.set_ylabel("Posizione Z [m]"); ax_pz.set_xlabel("Tempo [s]"); ax_pz.grid(True)

    # --- FIGURA 3: Velocità nel Tempo (Vx e Vz separate) ---
    fig3, (ax_vx, ax_vz) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig3.suptitle("Andamento Velocità (Limite Rete ±1m/s)", fontsize=14)
    ax_vx.set_ylabel("Velocità X [m/s]"); ax_vx.grid(True)
    ax_vz.set_ylabel("Velocità Z [m/s]"); ax_vz.set_xlabel("Tempo [s]"); ax_vz.grid(True)
    
    # Linee di sicurezza per la Rete Neurale
    for ax in [ax_vx, ax_vz]:
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.7)
        ax.axhline(-1.0, color='k', linestyle=':', alpha=0.7)

    # --- FIGURA 4: Comandi Motori (U1 e U2 separati) ---
    fig4, (ax_u1, ax_u2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig4.suptitle("Comando Motori (Spinta)", fontsize=14)
    ax_u1.set_ylabel("Motore 1 [$u^2$]"); ax_u1.grid(True)
    ax_u2.set_ylabel("Motore 2 [$u^2$]"); ax_u2.set_xlabel("Tempo [s]"); ax_u2.grid(True)
    ax_u1.axhline(model.u_bar, color='r', linestyle='--', alpha=0.5, label='Max Power')
    ax_u2.axhline(model.u_bar, color='r', linestyle='--', alpha=0.5, label='Max Power')

    # Popoliamo tutti i grafici con il loop sui risultati
    for res, color in zip(risultati, colors):
        nome = res['nome']
        x_h = res['x_hist']
        u_h = res['u_hist']
        a_h = res['alpha_hist']
        xref = res['xref']
        
        # Fig 1
        ax1.plot(x_h[:, 0], x_h[:, 1], color=color, linewidth=2, label=nome)
        ax1.plot(xref[0], xref[1], 'X', color=color, markersize=10) 
        ax2.plot(time_u, a_h, color=color, linewidth=1.5, alpha=0.8, label=nome)

        # Fig 2 (Posizioni)
        ax_px.plot(time_x, x_h[:, 0], color=color, linewidth=2, label=nome)
        ax_px.axhline(xref[0], color=color, linestyle='--', alpha=0.3) # Linea target X
        ax_pz.plot(time_x, x_h[:, 1], color=color, linewidth=2, label=nome)
        ax_pz.axhline(xref[1], color=color, linestyle='--', alpha=0.3) # Linea target Z

        # Fig 3 (Velocità)
        ax_vx.plot(time_x, x_h[:, 3], color=color, linewidth=1.5)
        ax_vz.plot(time_x, x_h[:, 4], color=color, linewidth=1.5)

        # Fig 4 (Motori)
        ax_u1.plot(time_u, u_h[:, 0], color=color, linewidth=1.5)
        ax_u2.plot(time_u, u_h[:, 1], color=color, linewidth=1.5)

    # Aggiungiamo le legende in modo pulito
    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)
    ax_px.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8) # Legenda fuori dal grafico
    
    # Mostriamo tutte le finestre contemporaneamente!
    plt.show()

if __name__ == '__main__':
    main()
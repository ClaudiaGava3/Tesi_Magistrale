import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.append("/home/uav_mpc/Tesi_Claudia/src/DEEPREACH/deepreach")

os.chdir("/home/uav_mpc/Tesi_Claudia/src/DEEPREACH/deepreach")

from try_bicopter_set import evaluate_V as evaluate_deep
from try_bicopter_set import model as deep_model
from try_bicopter_set import dynamics_obj as deep_dynamics

# 2. IMPORTIAMO LA TUA RETE NEURALE
# Sostituisci "nome_del_tuo_file" con il file dove hai definito la classe NeuralNetwork
os.chdir("/home/uav_mpc/Tesi_Claudia")
from learning import NeuralNetwork

IS_CUBE = True

def main():
    # --- A. CARICAMENTO DATI ---
    if IS_CUBE:
        dataset_path = "data/sth_3D_TEST_dataset_classification_cube.npy" # Aggiorna col path corretto
        nn_filename = "nn_models/sth_gelu_3D_cube.pt"
        input_size_nn = 9
    else:
        dataset_path = "data/sth_3D_TEST_dataset_classification_randB.npy" # Aggiorna col path corretto
        nn_filename = "nn_models/sth_gelu_3D_randB.pt"
        input_size_nn = 15

    data = np.load(dataset_path)
    print(f"Dataset caricato: {data.shape[0]} campioni.")
    print(f"Modalità test: {'Cubo (9 inputs)' if IS_CUBE else 'Parallelepipedo (15 inputs)'}")

    # --- B. CARICAMENTO RETE NEURALE ---
    checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
    mean_X = torch.tensor(checkpoint['mean'], dtype=torch.float32)
    std_X = torch.tensor(checkpoint['std'], dtype=torch.float32)

    # Inizializza la tua rete (nota: input_size=8)
    net = NeuralNetwork(
        input_size=input_size_nn, #9 per cube #15 per general
        hidden_size=1024,
        output_size=1, 
        number_hidden=2,
        activation=torch.nn.GELU(approximate='tanh'), 
        ub=1
    )
    net.load_state_dict(checkpoint['model'])
    net.eval()

    # --- C. VARIABILI STATISTICHE ---
    VP_nn, FP_nn, VN_nn, FN_nn = 0, 0, 0, 0
    VP_dp, FP_dp, VN_dp, FN_dp = 0, 0, 0, 0

    # --- D. CICLO DI CONFRONTO ---
    for i in range(data.shape[0]):
        row = data[i]
        
        # ATTENZIONE: SOSTITUISCI GLI INDICI [0], [1] ECC. CON QUELLI CORRETTI DEL TUO DATASET!
        phi = row[0]
        theta = row[1]
        psi = row[2]
        vx = row[3]
        vy = row[4]
        vz = row[5]
        wx = row[6]
        wy = row[7]
        wz = row[8] # commentare da qua in poi per caso cube
        x_max = row[9]
        y_max = row[10]
        z_max = row[11]
        x_min = row[12]
        y_min = row[13]
        z_min = row[14]
        
        # Verdetto OCP (Ground Truth) -> Supponiamo sia l'ultima colonna
        ocp_success = row[-1] 

        # --- Test DeepReach ---
        deep_state = [0.0, 0.0, 0.0, phi,theta, psi, x_min, x_max, y_min, y_max,z_min, z_max, vx, vy, vz, wx, wy, wz]
        V_val = evaluate_deep(deep_model, deep_dynamics, deep_state, time=0.5)
        deep_is_safe = (V_val > 0)

        # --- Test Tua NN ---
        # 1. Calcolo box normalizzato solo per caso con 6 dimensioni diverse
        if IS_CUBE:
            norma = x_max # Essendo un cubo, tutti i lati sono identici
            # Prepariamo il tensore solo con le 9 cinematiche
            x_in = torch.tensor([phi, theta, psi, vx, vy, vz, wx, wy, wz] , dtype=torch.float32)
        else:
            # Calcolo norma del parallelepipedo
            norma = np.sqrt(x_min**2 + x_max**2 + y_min**2 + y_max**2 + z_min**2 + z_max**2)
            # Vettore normalizzato
            box_n = [x_max/norma, y_max/norma, z_max/norma, x_min/norma, y_min/norma, z_min/norma]
            # Prepariamo il tensore con le 9 cinematiche + i 6 lati normalizzati
            x_in = torch.tensor([phi, theta, psi, vx, vy, vz, wx, wy, wz]  + box_n, dtype=torch.float32)

        # 2. Normalizzazione
        x_norm = (x_in - mean_X) / std_X

        
        # 3. Predizione e controllo
        with torch.no_grad():
            alpha_pred = net(x_norm).item()
        
        nn_is_safe = (alpha_pred <= norma)

        # --- LOGICA DI CONFRONTO (Tua Rete Neurale) ---
        if not nn_is_safe and ocp_success == 0.0:
            VP_nn += 1  # Vero Positivo: Rete dice insicuro, OCP fallisce
        elif not nn_is_safe and ocp_success == 1.0:
            FP_nn += 1  # Falso Positivo: Rete dice insicuro, OCP si salva
        elif nn_is_safe and ocp_success == 1.0:
            VN_nn += 1  # Vero Negativo: Rete dice sicuro, OCP si salva
        elif nn_is_safe and ocp_success == 0.0:
            FN_nn += 1  # Falso Negativo: Rete dice sicuro, OCP fallisce (CRITICO!)

        # --- LOGICA DI CONFRONTO (DeepReach) ---
        if not deep_is_safe and ocp_success == 0.0:
            VP_dp += 1
        elif not deep_is_safe and ocp_success == 1.0:
            FP_dp += 1
        elif deep_is_safe and ocp_success == 1.0:
            VN_dp += 1
        elif deep_is_safe and ocp_success == 0.0:
            FN_dp += 1

    print("Test completato senza errori di indice!")

    print("\n--- RISULTATI TUA RETE NEURALE ---")
    print(f"Veri Positivi (VP): {VP_nn}")
    print(f"Falsi Positivi (FP): {FP_nn}")
    print(f"Veri Negativi (VN): {VN_nn}")
    print(f"Falsi Negativi (FN): {FN_nn}  <-- I più pericolosi!")

    print("\n--- RISULTATI DEEPREACH ---")
    print(f"Veri Positivi (VP): {VP_dp}")
    print(f"Falsi Positivi (FP): {FP_dp}")
    print(f"Veri Negativi (VN): {VN_dp}")
    print(f"Falsi Negativi (FN): {FN_dp}")

if __name__ == '__main__':
    main()
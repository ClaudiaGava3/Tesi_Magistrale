
# ==========================================================================================================================================================
"""Compare RBOC and DeepReach safety predictions against OCP ground truth.

For each sampled state and box, the script evaluates the RBOC-trained neural
network and the DeepReach value function (2 different training), then compares their binary safety predictions with the OCP feasibility label. It reports confusion-matrix
counts for both models; the positive class is defined as unsafe.
"""
# ==========================================================================================================================================================

import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.append("/home/claudia/TESI/VBOC-3D/src/DEEPREACH/deepreach")
os.chdir("/home/claudia/TESI/VBOC-3D/src/DEEPREACH/deepreach")

from try_quadcopter_set import evaluate_V as evaluate_deep
from try_quadcopter_set import model as deep_model
from try_quadcopter_set import dynamics_obj as deep_dynamics

# Import the RBOC-trained neural network architecture.
os.chdir("/home/claudia/TESI/VBOC-3D/src/DEEPREACH")
from learning import NeuralNetwork

IS_CUBE = True

def main():
    # --- Load samples containing state, box dimensions, and OCP labels ---
    if IS_CUBE:
        dataset_path = "src/DEEPREACH/sth_TEST_dataset_classification2_5m_cube.npy"
        nn_filename = "nn_models/sth/sth_gelu_randB_cube.pt"
        input_size_nn = 9
    else:
        dataset_path = "src/DEEPREACH/sth_TEST_dataset_classification2_5m_randB.npy"
        nn_filename = "nn_models/sth/sth_gelu_randB.pt"
        input_size_nn = 15

    data = np.load(dataset_path)
    print(f"Dataset caricato: {data.shape[0]} campioni.")
    print(f"Modalità test: {'Cubo (9 inputs)' if IS_CUBE else 'Parallelepipedo (15 inputs)'}")

    # --- Load the trained network and its input-normalization statistics ---
    checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
    mean_X = torch.tensor(checkpoint['mean'], dtype=torch.float32)
    std_X = torch.tensor(checkpoint['std'], dtype=torch.float32)

    # The network input is [angles, vel, rot_vel] plus six normalized box sides in the assimetric case.
    net = NeuralNetwork(
        input_size=input_size_nn, #9 for cube 15 for assimetric box
        hidden_size=1024,
        output_size=1, 
        number_hidden=2,
        activation=torch.nn.GELU(approximate='tanh'), 
        ub=1
    )
    net.load_state_dict(checkpoint['model'])
    net.eval()

    # --- Confusion-matrix counts; unsafe predictions are the positive class ---
    VP_nn, FP_nn, VN_nn, FN_nn = 0, 0, 0, 0
    VP_dp, FP_dp, VN_dp, FN_dp = 0, 0, 0, 0
    contpos_nn, contneg_nn =0,0
    contpos_dp, contneg_dp =0,0
    

    dp_fp_boxes = []
    dp_fp_vels = []
    nn_fp_boxes = []
    nn_fp_vels = []
    

    # --- Evaluate both methods and compare with OCP feasibility ground truth ---
    for i in range(data.shape[0]):
        row = data[i]
        
        phi = row[0]
        theta = row[1]
        psi = row[2]
        vx = row[3]
        vy = row[4]
        vz = row[5]
        wx = row[6]
        wy = row[7]
        wz = row[8]
        x_max = row[9]
        y_max = row[10]
        z_max = row[11]
        x_min = row[12]
        y_min = row[13]
        z_min = row[14]
        
        # Ground-truth label: 1 means OCP feasible/safe; 0 means infeasible/unsafe.
        ocp_success = row[-1] 

        # --- DeepReach value-function sign: positive is classified as safe ---
        alpha = 0.0
        deep_state = [0.0, 0.0, 0.0, phi,theta, psi, vx, vy, vz, wx, wy, wz, x_max]
        V_val = evaluate_deep(deep_model, deep_dynamics, deep_state, time=0.6)
        deep_is_safe = (V_val > alpha)

        # --- Test NN ---
        # 1. Recover the common box scale and normalize the six side dimensions.
        if IS_CUBE:
            norma = x_max # For a cube all the sides are identical!
            x_in = torch.tensor([phi, theta, psi, vx, vy, vz, wx, wy, wz] , dtype=torch.float32)
        else:
            norma = np.sqrt(x_min**2 + x_max**2 + y_min**2 + y_max**2 + z_min**2 + z_max**2)
            box_n = [x_max/norma, y_max/norma, z_max/norma, x_min/norma, y_min/norma, z_min/norma]
            x_in = torch.tensor([phi, theta, psi, vx, vy, vz, wx, wy, wz]  + box_n, dtype=torch.float32)

        # Normalization
        x_norm = (x_in - mean_X) / std_X

        beta=0.0

        # 3. Predict the required scale; classify safe if it fits within the box.
        with torch.no_grad():
            alpha_pred = net(x_norm).item()
        
        nn_is_safe = (alpha_pred <= (1-beta)*norma)

        if nn_is_safe:
            contpos_nn += 1
        else:
            contneg_nn += 1

        # --- NN confusion matrix (positive = unsafe; OCP label is ground truth) ---
        if not nn_is_safe and ocp_success == 0.0:
            VP_nn += 1  # True Positive: NN predicts unsafe ocp fails
        elif not nn_is_safe and ocp_success == 1.0:
            FP_nn += 1  # False Positive: NN predicts unsafe ocp succeeds
            # nn_fp_boxes.append(x_max)
            # nn_fp_vels.append(np.sqrt(vx**2 + vy**2 + vz**2))
        elif nn_is_safe and ocp_success == 1.0:
            VN_nn += 1   # True Negative: NN predicts safe ocp succeeds
        elif nn_is_safe and ocp_success == 0.0:
            FN_nn += 1  # False Negative: NN predicts safe ocp fails (CRITICAL!)

        
        if deep_is_safe:
            contpos_dp += 1
        else:
            contneg_dp += 1

        # --- DeepReach confusion matrix with the same class convention ---        
        if not deep_is_safe and ocp_success == 0.0:
            VP_dp += 1
        elif not deep_is_safe and ocp_success == 1.0:
            FP_dp += 1
            # dp_fp_boxes.append(x_max)
            # dp_fp_vels.append(np.sqrt(vx**2 + vy**2 + vz**2))
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
    # print(f"Negativi rete: {contneg_nn}")
    # print(f"Positivi rete: {contpos_nn}")

    # print("\n--- IDENTIKIT FALSI POSITIVI NN ---")
    # if FP_nn > 0:
    #     print(f"Dimensione media Box nei Falsi Positivi: {np.mean(nn_fp_boxes):.2f} metri")
    #     print(f"Velocità media nei Falsi Positivi: {np.mean(nn_fp_vels):.2f} m/s")
    #     print(f"Tasso di Falso Allarme: {(FP_nn / (VP_nn + FP_nn) * 100):.1f}%")


    print("\n--- RISULTATI DEEPREACH ---")
    print(f"Veri Positivi (VP): {VP_dp}")
    print(f"Falsi Positivi (FP): {FP_dp}")
    print(f"Veri Negativi (VN): {VN_dp}")
    print(f"Falsi Negativi (FN): {FN_dp}")
    # print(f"Negativi rete: {contneg_dp}")
    # print(f"Positivi rete: {contpos_dp}")

    # print("\n--- IDENTIKIT FALSI POSITIVI DEEPREACH ---")
    # if FP_dp > 0:
    #     print(f"Dimensione media Box nei Falsi Positivi: {np.mean(dp_fp_boxes):.2f} metri")
    #     print(f"Velocità media nei Falsi Positivi: {np.mean(dp_fp_vels):.2f} m/s")
    #     print(f"Tasso di Falso Allarme: {(FP_dp / (VP_dp + FP_dp) * 100):.1f}%")



if __name__ == '__main__':
    main()
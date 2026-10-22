
# ==========================================================================================================================================================
"""Compare RBOC and DeepReach safety predictions against OCP ground truth.

For each sampled state and box, the script evaluates the RBOC-trained neural
network and the DeepReach value function, then compares their binary safety
predictions with the OCP feasibility label. It reports confusion-matrix
counts for both models; the positive class is defined as unsafe.
"""
# ==========================================================================================================================================================

import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.append("/home/claudia/TESI/VBOC-2D/src/DEEPREACH/deepreach")
os.chdir("/home/claudia/TESI/VBOC-2D/src/DEEPREACH/deepreach")

from try_bicopter_set import evaluate_V as evaluate_deep
from try_bicopter_set import model as deep_model
from try_bicopter_set import dynamics_obj as deep_dynamics

# Import the RBOC-trained neural network architecture.
os.chdir("/home/claudia/TESI/VBOC-2D/src/DEEPREACH")
from learning import NeuralNetwork

def main():
    # --- Load samples containing state, box dimensions, and OCP labels ---
    dataset_path = "src/DEEPREACH/sth_TEST_dataset_classification2_5m.npy"
    data = np.load(dataset_path)
    print(f"Dataset caricato: {data.shape[0]} campioni.")

    # --- Load the trained network and its input-normalization statistics ---
    nn_filename = "nn_models/sth/sth_gelu_randB.pt"
    checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
    mean_X = torch.tensor(checkpoint['mean'], dtype=torch.float32)
    std_X = torch.tensor(checkpoint['std'], dtype=torch.float32)

    # The network input is [theta, vx, vz, wy] plus four normalized box sides.
    net = NeuralNetwork(
        input_size=8, 
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

    # --- Evaluate both methods and compare with OCP feasibility ground truth ---
    for i in range(data.shape[0]):
        row = data[i]
        
        theta = row[0]
        vx = row[1]
        vz = row[2]
        wy = row[3]
        x_max = row[4]
        z_max = row[5]
        x_min = row[6]
        z_min = row[7]
        
        # Ground-truth label: 1 means OCP feasible/safe; 0 means infeasible/unsafe.
        ocp_success = row[-1] 

        # --- DeepReach value-function sign: positive is classified as safe ---
        alpha = 0.0
        deep_state = [0.0, 0.0, theta, x_min, x_max, z_min, z_max, vx, vz, wy]
        V_val = evaluate_deep(deep_model, deep_dynamics, deep_state, time=0.5)
        deep_is_safe = (V_val > alpha)

        # --- Test NN ---
        # 1. Recover the common box scale and normalize the four side dimensions.
        norma = np.sqrt(x_min**2 + x_max**2 + z_min**2 + z_max**2)
        box_n = [x_max/norma, z_max/norma, x_min/norma, z_min/norma]
        
        # 2. Prepare and standardize the eight network input features.
        x_in = torch.tensor([theta, vx, vz, wy] + box_n, dtype=torch.float32)
        x_norm = (x_in - mean_X) / std_X
        
        # 3. Predict the required scale; classify safe if it fits within the box.
        with torch.no_grad():
            alpha_pred = net(x_norm).item()

        beta = 0.0
        
        nn_is_safe = (alpha_pred <= (1-beta)*norma)

        # --- NN confusion matrix (positive = unsafe; OCP label is ground truth) ---
        if not nn_is_safe and ocp_success == 0.0:
            VP_nn += 1  # True Positive: NN predicts unsafe ocp fails
        elif not nn_is_safe and ocp_success == 1.0:
            FP_nn += 1  # False Positive: NN predicts unsafe ocp succeeds
        elif nn_is_safe and ocp_success == 1.0:
            VN_nn += 1  # True Negative: NN predicts safe ocp succeeds
        elif nn_is_safe and ocp_success == 0.0:
            FN_nn += 1  # False Negative: NN predicts safe ocp fails (CRITICAL!)

        # --- DeepReach confusion matrix with the same class convention ---
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
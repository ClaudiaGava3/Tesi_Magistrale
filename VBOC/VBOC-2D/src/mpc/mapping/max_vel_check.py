import numpy as np
import os

# Sostituisci con il percorso corretto se diverso
DATA_DIR = "data/"
robotic_system = 'sth'

print("=== CARICAMENTO DATASET ===\n")
x_data = np.load(f"{DATA_DIR}{robotic_system}_x_vboc_randB.npy")
b_data = np.load(f"{DATA_DIR}{robotic_system}_b_vboc_randB.npy")

print(f"Dimensioni totali x_data (Stati completi): {x_data.shape}")
print(f"Dimensioni totali b_data (Target alpha): {b_data.shape}")
print("-" * 50)

for i in range(5):
    stato = x_data[i]
    b_val = b_data[i]
    
    # Il box normalizzato si trova agli indici 6, 7, 8, 9 di x_data
    box_norm = stato[6:10]
    
    print(f"📌 CAMPIONE {i+1}:")
    print(f"  Posizione [x, z]:     [{stato[0]:.3f}, {stato[1]:.3f}] m")
    print(f"  Angolo [theta]:       {stato[2]:.3f} rad")
    print(f"  Velocità [vx, vz]:    [{stato[3]:.3f}, {stato[4]:.3f}] m/s")
    print(f"  Velocità ang. [wy]:   {stato[5]:.3f} rad/s")
    
    # Stampiamo il box e l'alpha
    print(f"  Box normalizzato:     {np.round(box_norm, 3)}")
    
    # Gestiamo l'estrazione sicura di alpha (sia che sia un array [alpha] o un float)
    if isinstance(b_val, np.ndarray) and b_val.size == 1:
        print(f"  Fattore di Scala (α): {b_val[0]:.3f} m")
    else:
        print(f"  Fattore di Scala (α): {float(b_val):.3f} m")
        
    print("-" * 50)
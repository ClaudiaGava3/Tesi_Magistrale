import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def mostra_schizzo_target():
    # 1. Definizione Ostacoli
    poligoni = [
        # Basi fisse (Pavimento, Soffitto, Muretto finale)
        [[-2.0, -4.0], [25.0, -4.0], [25.0, -5.0], [-2.0, -5.0]], 
        [[-2.0,  5.0], [25.0,  5.0], [25.0,  6.0], [-2.0,  6.0]], 
        
        # Ostacoli Grigi (mantenuti)
        [[3.0, 1.0], [5.0, 3.0], [6.0, 1.0], [5.0, 0.0]],         # Rombo SX
        [[7.0, -3.0], [9.0, -3.0], [9.0, -0.5], [7.0, -0.5]],     # Quadrato Basso
        
        # NUOVI OSTACOLI BLU
        [[7.8, 3.9], [10.0, 3.9], [9.6, 0.8]],                     # Triangolo alto SX
        [[11.0, -0.4], [13.9, -0.6], [14.3, -1.7], [11.9, -2.8]], # Rettangolo obliquo basso
        [[12.0, 2.5], [12.6, 3.3], [13.6, 3.3], [14.1, 2.4], 
         [14.1, 1.2], [13.4, 0.9], [12.4, 1.0]],                  # Esagono centrale
        [[16.3, 4.1], [19.3, 4.2], [19.3, 2.5], [16.3, 2.5]],     # Rettangolo alto DX
        [[15.0, -1.0], [18.0, -2.0], [19.0, 0.0], [16.0, 1.0]],   # Rettangolo basso dx
    ]
    
    # Muretto finale a DX
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            segments.append([poli[i], poli[(i + 1) % n]])
    segments.append([[21.0, -4.0], [21.0, -2.0]])
            
    # I PUNTINI VERDI (Target Testati estratti dall'immagine)
    targets = [
        np.array([4.8, -2.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([6.2,  3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([7.6,  0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([10.2,-2.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.2, 4.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.1, 3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([14.8, 0.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.5,-2.6, 0.0, 0.0, 0.0, 0.0]),
        np.array([18.2, 1.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([19.6,-2.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([20.1, 3.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([22.5, 0.4, 0.0, 0.0, 0.0, 0.0])
    ]
    

    # 3. Plot
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    for poli in poligoni:
        poly = patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5)
        ax.add_patch(poly)
        
    for i, t in enumerate(targets):
        ax.scatter(t[0], t[1], color='red', marker='X', s=100, zorder=5, 
                   label='Target' if i==0 else "")
        
    ax.scatter(0.0, 0.0, color='blue', s=150, zorder=6, label='Punto di Partenza')
    
    plt.xlim(-2, 25)
    plt.ylim(-6, 6)
    plt.title('Bozza ambiente e target')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mostra_schizzo_target()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

# =============================================================================
# 1. FUNZIONI DI CONFIGURAZIONE AMBIENTE (Simili alle tue)
# =============================================================================
def get_ambiente_mock():
    """Simula l'ambiente 'Stanza 1' e una traiettoria per il test."""
    # Stanza 1: Corridoio (Sopra e sotto liberi) - Come in image_0.png
    obstacles = [
        [1.5, 4.5, 1.0, 3.0],    # Soffitto
        [1.5, 4.5, -5.0, -3.5]   # Pavimento
        # [x_min, x_max, z_min, z_max]
    ]
    x_start = np.array([0.0, 0.0])
    x_target = np.array([5.0, -1.5])
    
    # Simula una traiettoria a 37 step per avere un box per ogni step
    # x_hist: Array Nx2 di posizioni (X, Z)
    x_hist = np.array([
        [0.00,  0.00], [0.15, -0.05], [0.35, -0.15], [0.58, -0.32], [0.85, -0.55],
        [1.10, -0.82], [1.32, -1.10], [1.55, -1.35], [1.75, -1.58], [1.98, -1.78],
        [2.22, -1.82], [2.48, -1.85], [2.72, -1.80], [2.95, -1.72], [3.20, -1.65],
        [3.45, -1.58], [3.68, -1.53], [3.90, -1.50], [4.12, -1.48], [4.35, -1.48],
        [4.55, -1.48], [4.78, -1.48], [4.95, -1.48], [5.08, -1.48], [5.18, -1.48],
        [5.22, -1.48], [5.25, -1.48], [5.28, -1.48], [5.30, -1.48], [5.30, -1.48],
        [5.30, -1.48], [5.30, -1.48], [5.30, -1.48], [5.30, -1.48], [5.30, -1.48],
        [5.30, -1.48], [5.30, -1.48]
    ])
    
    # Simula i box assoluti corrispondenti a ogni step di x_hist
    # box_hist: Array N-1x4 di [xmin, xmax, zmin, zmax] assoluti.
    # Nota: L'ultimo step non ha un box (o il drone si è fermato).
    box_hist = []
    
    for t in range(len(x_hist) - 1):
        cx, cz = x_hist[t]
        
        # Simula l'algoritmo Max: il box si espande verso il target e si contrae vicino agli ostacoli
        if t < 10:
            # All'inizio, area aperta
            b_xmin, b_xmax, b_zmin, b_zmax = cx - 1.0, cx + 1.5, cz - 10.0, cz + 10.0
        elif t < 22:
            # Nel corridoio, stretto in Z
            b_xmin, b_xmax, b_zmin, b_zmax = cx - 0.5, cx + 1.2, cz - 1.0, cz + 0.8
        else:
            # Vicino al target, area aperta di nuovo
            b_xmin, b_xmax, b_zmin, b_zmax = cx - 0.5, cx + 0.8, cz - 5.0, cz + 5.0
            
        # Assicurati che i box non compenetrino gli ostacoli (mock statico)
        b_zmin = max(b_zmin, -3.5) # Pavimento
        b_zmax = min(b_zmax, 1.0)  # Soffitto
        if cx > 1.0 and cx < 5.0:
             b_xmax = min(b_xmax, 4.5) # Ostacolo finale in X
             
        box_hist.append([b_xmin, b_xmax, b_zmin, b_zmax])
        
    box_hist = np.array(box_hist)
    return obstacles, x_hist, box_hist, x_target, x_start

# =============================================================================
# 2. LOGICA DI ANIMAZIONE
# =============================================================================
def create_retrace_animation(filename='robot_retrace.mp4'):
    # Ottieni i dati mock simulati
    obstacles, x_hist, box_hist, x_target, x_start = get_ambiente_mock()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- Elementi Statici (sfondo) ---
    # Ostacoli
    for obs in obstacles:
        x_min, x_max, z_min, z_max = obs
        rect = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, 
                                 linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7, zorder=1)
        ax.add_patch(rect)
        
    # Start e Target
    ax.scatter(x_start[0], x_start[1], color='green', s=150, label='Start', zorder=5)
    ax.scatter(x_target[0], x_target[1], color='red', s=150, label='Target', marker='X', zorder=5)
    
    # La Traiettoria Completa (linea blu di sfondo)
    ax.plot(x_hist[:, 0], x_hist[:, 1], color='blue', linewidth=1.5, alpha=0.4, label='Path to Retrace', zorder=3)
    
    # --- Elementi Dinamici (da aggiornare in ogni frame) ---
    # Il Box Asimmetrico corrente (Verde trasparente)
    # Lo inizializziamo vuoto e lo aggiorneremo
    current_box_patch = patches.Rectangle((0, 0), 0, 0, linewidth=2, 
                                          edgecolor='green', facecolor='lime', alpha=0.2, zorder=2)
    ax.add_patch(current_box_patch)
    
    # Il Drone corrente (un punto blu che si muove)
    current_drone_scatter = ax.scatter([], [], color='blue', s=100, label='Drone', zorder=10)
    
    # --- Impostazioni Assi ---
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-6, 4])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    title = ax.set_title("Robot Retracing Trajectory (Step: 0)")
    
    # --- Funzione di Aggiornamento per ogni Frame ---
    def update(frame):
        # f: indice del frame (t)
        
        # 1. Ottieni la posizione del drone al tempo t
        drone_pos = x_hist[frame]
        
        # 2. Aggiorna il market del drone (deve essere una lista di liste per scatter)
        current_drone_scatter.set_offsets([drone_pos])
        
        # 3. Ottieni il box asimmetrico al tempo t (se disponibile)
        if frame < len(box_hist):
            b_xmin, b_xmax, b_zmin, b_zmax = box_hist[frame]
            # Aggiorna la geometria del patch esistente
            current_box_patch.set_xy((b_xmin, b_zmin))
            current_box_patch.set_width(b_xmax - b_xmin)
            current_box_patch.set_height(b_zmax - b_zmin)
            current_box_patch.set_visible(True) # Assicuriamoci che sia visibile
        else:
            # All'ultimo step, il box potrebbe non essere definito. Nascondilo.
            current_box_patch.set_visible(False)
            
        # 4. Aggiorna il titolo con il frame corrente
        title.set_text(f"Robot Retracing Trajectory (Step: {frame})")
        
        # Restituisce gli oggetti che sono stati modificati per un'ottimizzazione del disegno (blitting)
        return current_drone_scatter, current_box_patch, title

    # --- Crea l'Animazione ---
    num_frames = len(x_hist)
    # num_frames è il numero totale di step della traiettoria
    # interval: tempo in millisecondi tra un frame e l'altro (es: 100ms = 10 fps)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=120, blit=True)
    
    # --- Salva l'Animazione come Video ---
    try:
        # Se hai FFMpeg installato (necessario per .mp4), usa FFMpegWriter
        writer = FFMpegWriter(fps=8, metadata=dict(artist='AI Trajectory Animator'), bitrate=1800)
        ani.save(filename, writer=writer)
        print(f"✅ Video salvato con successo come '{filename}'")
    except (RuntimeError, ValueError) as e:
        # Altrimenti, prova a salvare una GIF
        gif_filename = filename.replace('.mp4', '.gif')
        ani.save(gif_filename, writer='pillow', fps=8)
        print(f"⚠️ Errore nel salvataggio video (manca FFMpeg?). Salvata GIF come '{gif_filename}'")
    except Exception as e:
        print(f"❌ Errore sconosciuto nel salvataggio: {e}")

    plt.close(fig)

if __name__ == '__main__':
    print("--- Generazione Animazione del Robot in corso ---")
    create_retrace_animation('robot_retrace.mp4')
    print("--- Finito! Puoi vedere il video nella stessa cartella. ---")
# RISULTATI CAMBIANDO ORIZZONTE, RAGGIO LIDAR E MARGINE

# HO PROVATO A AUMENTARE E ABBASSARE RISOLUZIONE PER EVITARE GLITCH, NON CAMBIA UN CAZZO, ha un problema se arriva troppo vicino agli ostacoli!

#senza margine (ws_modificato (directional))

# LD1.5
# N10; 12,1,0
# N15: 11,2,0
# N20: 9,4,0
# N25: 11,2,0
# N30: 8,5,0

# LD 3.0
# N10: 11,1,1
# N15: 8,0,5
# N20: 9,0,4
# N25: 9,3,1
# N30: 9,4,0

# LD 4.5
# N10: 7,4,2
# N15: 1,0,12
# N20: 5,0,8
# N25: 9,2,2        12 da timout ma è fallimento
# N30: 7,4,2

#######################################

# NON USARE PER STUDIO!!!!

# margine +15 =traslazione box a dx (video cancellati!)
#LD 1.5
#N10: 12,1,0
#N15:
#N20: 11,2,0
#N25:
#N30: 7,6,0

# LD 3.0
# N10: 13,0,0
# N15
# N20: 11,2,0
# N25
# N30: 8,4,1

# LD 4.5
# N10: 10,3,0
# N15
# N20: 10,2,1
# N25
# N30: 7,4,2

#######################################

# margine +-15 ingrandimento box, correzione (ws_modificato_1)
# LD1.5
# N10: 12,1,0
# N15: 11,2,0
# N20: 11,2,0
# N25: 11,2,0
# N30: 7,6,0

#LD 3.0
#N10: 12,1,0
#N15: 9,2,2      target 5 da timeout ma è un fallimento
#N20: 10,2,1 (11,2,0?)
#N25: 10,2,1
#N30: 8,4,1      target 10 da timeout ma è fallimento

#LD 4.5
#N10: 11,2,0
#N15: 3,1,9
#N20: 10,2,1
#N25: 11,1,1
#N30: 8,3,2 


# margine 15 non va tanto bene, provo con margine 10, non va meglio, metto + 15 solo per max non va meglio

#######################################

# margine 20 (ws_modificato_2)
# Lidar1.5
# N10: 8,3,2
# N15: 11,2,0
# N20: 11,2,0
# N25: 11,2,0
# N30: 7,5,1         target 10 da timeout ma è fallimento

# Lidar3.0
# N10: 13,0,0
# N15: 11,2,0 
# N20: 9,1,3
# N25: 11,2,0
# N30: 7,2,4         target 10,12 da timeout ma è fallimento

# Lidar4.5
# N10: 12,0,1
# N15: 11,2,0
# N20: 10,2,1        target 11 fa fallimento, manca video
# N25: 9,2,2         target 10 fa fallimento, manca video
# N30: 8,3,2

############################################################################################################################################################################################################################################

# ============================================================================================================================================================

# RISULTATI TEST con tempi computazionali


# ======================================================================
# ⏱  STATISTICHE TEMPI COMPUTAZIONALI (N=10, LiDAR=3.0m)
# ======================================================================
# con 100,100 iter da mio computer con creazione box
# Tempo Medio:     0.27739 secondi
# 99-Percentile:   0.71477 secondi
# Tempo Massimo:   1.87601 secondi
# con 100,100 iter da ws con creazione box
# Tempo Medio:     0.06509 secondi
# 99-Percentile:   0.15545 secondi
# Tempo Massimo:   0.17887 secondi
# con 100,100 iter da ws senza creazione box
# Tempo Medio:     0.05651 secondi
# 99-Percentile:   0.14705 secondi
# Tempo Massimo:   0.17314 secondi
# ws con RTI
# Tempo Medio:     0.05498 secondi
# 99-Percentile:   0.14279 secondi
# Tempo Massimo:   0.17044 secondi
# ws con Gauss_Newton
# Tempo Medio:     0.05731 secondi
# 99-Percentile:   0.14905 secondi
# Tempo Massimo:   0.17420 secondi
# ws con RTI e GAUSS_NEWTON
# Tempo Medio:     0.05553 secondi
# 99-Percentile:   0.14415 secondi
# Tempo Massimo:   0.17111 secondi
# RTI non si era attivato, con 10,10 iterazioni da sti tempi (12,1,0)
# Tempo Medio:     0.01122 secondi
# 99-Percentile:   0.01355 secondi
# Tempo Massimo:   0.04332 secondi
# con 5,100 iterazioni (13,0,0)
# Tempo Medio:     0.00749 secondi
# 99-Percentile:   0.00915 secondi
# Tempo Massimo:   0.04374 secondi
# ======================================================================
# ⏱  STATISTICHE TEMPI COMPUTAZIONALI (N=15, LiDAR=3.0m)
# ======================================================================
# con 100,100 iter da mio computer con creazione box
# Tempo Medio:     0.57418 secondi
# 99-Percentile:   0.81368 secondi
# Tempo Massimo:   1.75915 secondi
# con 100,100 iter da ws con creazione box
# Tempo Medio:     0.13900 secondi
# 99-Percentile:   0.17855 secondi
# Tempo Massimo:   0.21331 secondi
# con 100,100 iter da ws senza creazione box
# Tempo Medio:     0.13191 secondi
# 99-Percentile:   0.17221 secondi
# Tempo Massimo:   0.19115 secondi
# con 5,100 iterazioni (9,2,2, 5 da time ma è fall, 8 da succ ma è a filo)
# Tempo Medio:     0.00903 secondi
# 99-Percentile:   0.01096 secondi
# Tempo Massimo:   0.04402 secondi
# ======================================================================
# ⏱  STATISTICHE TEMPI COMPUTAZIONALI (N=20, LiDAR=3.0m)
# ======================================================================
# con 100,100 iter damio computer con creazione box
# Tempo Medio:     0.68461 secondi
# 99-Percentile:   1.25167 secondi
# Tempo Massimo:   4.31679 secondi
# con 100,100 iter da ws  con creazione box
# Tempo Medio:     0.15969 secondi
# 99-Percentile:   0.19836 secondi
# Tempo Massimo:   0.22622 secondi
# con 100,100 iter da ws senza creazione box
# Tempo Medio:     0.15614 secondi
# 99-Percentile:   0.19590 secondi
# Tempo Massimo:   0.24720 secondi
# RTI non si era attivato, con 10 iterazioni da sti tempi (12,1,0)
# Tempo Medio:     0.01385 secondi
# 99-Percentile:   0.01597 secondi
# Tempo Massimo:   0.04303 secondi
# con 5,100 iterazioni (11,2,0)
# Tempo Medio:     0.01008 secondi
# 99-Percentile:   0.01209 secondi
# Tempo Massimo:   0.04577 secondi
# ======================================================================
# ⏱  STATISTICHE TEMPI COMPUTAZIONALI (N=25, LiDAR=3.0m)
# ======================================================================
# con 100,100 iter da ws  con creazione box
# Tempo Medio:     0.17092 secondi
# 99-Percentile:   0.22096 secondi
# Tempo Massimo:   0.25992 secondi
# con 100,100 iter da ws senza creazione box
# Tempo Medio:     0.16488 secondi
# 99-Percentile:   0.21432 secondi
# Tempo Massimo:   0.25432 secondi  
# con 5,100 iterazioni (10,2,1, 11 da time ma è fail) senza creazione box
# Tempo Medio:     0.01108 secondi
# 99-Percentile:   0.01332 secondi
# Tempo Massimo:   0.04291 secondi
# ======================================================================
# ⏱  STATISTICHE TEMPI COMPUTAZIONALI (N=30, LiDAR=3.0m)
# ======================================================================  
# ws senza creazione box
# Tempo Medio:     0.14489 secondi
# 99-Percentile:   0.22985 secondi
# Tempo Massimo:   0.27546 secondi
# con 5,100 iterazioni (7,1,5)
# Tempo Medio:     0.01188 secondi
# 99-Percentile:   0.01439 secondi
# Tempo Massimo:   0.07822 secondi
# ======================================================================

# Lidar 3.0
# risultati con 100,100 iterazioni (come test) su workstation
# senza creazione box
# N10 (12,1,0)
# Tempo Medio:     0.05651 secondi
# 99-Percentile:   0.14705 secondi
# Tempo Massimo:   0.17314 secondi
# N15 (9,2,2)
# Tempo Medio:     0.13191 secondi
# 99-Percentile:   0.17221 secondi
# Tempo Massimo:   0.19115 secondi
# N20 (10,2,1) (11,2,0 su tesi)
# Tempo Medio:     0.15614 secondi
# 99-Percentile:   0.19590 secondi
# Tempo Massimo:   0.24720 secondi
# N25 (10,2,1)  
# Tempo Medio:     0.16488 secondi
# 99-Percentile:   0.21432 secondi
# Tempo Massimo:   0.25432 secondi  
# N30 (8,4,1)
# Tempo Medio:     0.14489 secondi
# 99-Percentile:   0.22985 secondi
# Tempo Massimo:   0.27546 secondi

# risultati con 5,100 iterazioni (diverso da test) su workstation
# senza creazione box
# N10 (13,0,0)
# Tempo Medio:     0.00749 secondi
# 99-Percentile:   0.00915 secondi
# Tempo Massimo:   0.04374 secondi
# N15 (9,2,2)
# Tempo Medio:     0.00903 secondi
# 99-Percentile:   0.01096 secondi
# Tempo Massimo:   0.04402 secondi
# N20 (11,2,0)
# Tempo Medio:     0.01008 secondi
# 99-Percentile:   0.01209 secondi
# Tempo Massimo:   0.04577 secondi
# N25 (10,2,1)
# Tempo Medio:     0.01108 secondi
# 99-Percentile:   0.01332 secondi
# Tempo Massimo:   0.04291 secondi
# N30 (7,1,5)
# Tempo Medio:     0.01188 secondi
# 99-Percentile:   0.01439 secondi
# Tempo Massimo:   0.07822 secondi

# con creazione box
# N10 (13,0,0)
# con ris 360 ma metà dei raggi, 5,100 (11,2,0) ----> VERSIONE FINALE
# Tempo Medio:     0.01346 secondi
# 99-Percentile:   0.01581 secondi
# Tempo Massimo:   0.03343 secondi
# N15 (9,2,2)
# con ris 360 ma metà dei raggi, 5,100 (11,2,0) ----> VERSIONE FINALE
# Tempo Medio:     0.01516 secondi
# 99-Percentile:   0.01771 secondi
# Tempo Massimo:   0.08835 secondi
# N20 (11,2,0)
# con 5,100 iterazioni (11,2,0)
# Tempo Medio:     0.02056 secondi
# 99-Percentile:   0.02316 secondi
# Tempo Massimo:   0.05397 secondi
# abbassato raggio lidar a 180 (11,2,0)
# Tempo Medio:     0.01696 secondi
# 99-Percentile:   0.02133 secondi
# Tempo Massimo:   0.04980 secondi
# Runge Kutta secondo ordine (11,2,0)
# Tempo Medio:     0.02086 secondi
# 99-Percentile:   0.02468 secondi
# Tempo Massimo:   0.04958 secondi
# GAUSS_NEWTON
# Tempo Medio:     0.02138 secondi
# 99-Percentile:   0.02621 secondi
# Tempo Massimo:   0.05212 secondi
# solver SPEED (11,1,1) (SPEED_ABS cambiava tanto successi)
# Tempo Medio:     0.02120 secondi
# 99-Percentile:   0.02504 secondi
# Tempo Massimo:   0.05370 secondi
# con 4,100 iterazioni (11,1,1)
# Tempo Medio:     0.01916 secondi
# 99-Percentile:   0.02139 secondi
# Tempo Massimo:   0.08918 secondi
# con speed abs 5,100 (9,2,2)
# Tempo Medio:     0.02041 secondi
# 99-Percentile:   0.02276 secondi
# Tempo Massimo:   0.09171 secondi
# con speed e t>0 (10,2,1)
# Tempo Medio:     0.01991 secondi
# 99-Percentile:   0.02247 secondi
# Tempo Massimo:   0.03236 secondi
# con 3,100 e t>0 + ROBUST (11,1,1)
# Tempo Medio:     0.01772 secondi
# 99-Percentile:   0.02069 secondi
# Tempo Massimo:   0.03403 secondi
# con nuova funzione lidar e 5,100
# Tempo Medio:     0.01155 secondi
# 99-Percentile:   0.01364 secondi
# Tempo Massimo:   0.01969 secondi
# con nuova funzione lidar e 10,100
# Tempo Medio:     0.01954 secondi
# 99-Percentile:   0.02312 secondi
# Tempo Massimo:   0.03018 secondi
# con 1,100 (10,2,1)
# Tempo Medio:     0.01397 secondi
# 99-Percentile:   0.01843 secondi
# Tempo Massimo:   0.08630 secondi
# con ris 360 ma metà dei raggi, 5,100 (11,2,0) ----> VERSIONE FINALE
# Tempo Medio:     0.01620 secondi
# 99-Percentile:   0.01861 secondi
# Tempo Massimo:   0.02610 secondi
# N25 (10,2,1)
# con ris 360 ma metà dei raggi, 5,100 (11,2,0) ----> VERSIONE FINALE
# Tempo Medio:     0.01759 secondi
# 99-Percentile:   0.02149 secondi
# Tempo Massimo:   0.08887 secondi
# N30 (7,0,6)
# con ris 360 ma metà dei raggi, 5,100 (11,2,0) ----> VERSIONE FINALE
# Tempo Medio:     0.01809 secondi
# 99-Percentile:   0.02669 secondi
# Tempo Massimo:   0.21881 secondi


# ======================================================================


# seed 44
# ======================================================================
# 📊 RESOCONTO GLOBALE RECURSIVE FEASIBILITY (Traiettoria k in Box k+1)
# ======================================================================
# Target 01 | Esito: Successes | DistTot: 44.27m| DistX: 44.26m | Costo: 1768.20  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 02 | Esito: Successes | DistTot: 41.64m| DistX: 41.42m | Costo: 6792.16  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 03 | Esito: Successes | DistTot: 50.43m| DistX: 50.38m | Costo: 6079.78  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 04 | Esito: Successes | DistTot: 46.28m| DistX: 46.26m | Costo: 1941.22  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 05 | Esito: Successes | DistTot: 49.91m| DistX: 49.89m | Costo: 2502.76  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 06 | Esito: Successes | DistTot: 47.46m| DistX: 46.26m | Costo: 21201.75 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 07 | Esito: Successes | DistTot: 39.84m| DistX: 39.79m | Costo: 1451.16  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 08 | Esito: Successes | DistTot: 41.29m| DistX: 41.28m | Costo: 1540.94  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 09 | Esito: Successes | DistTot: 45.28m| DistX: 45.26m | Costo: 1865.81  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 10 | Esito: Successes | DistTot: 48.74m| DistX: 48.74m | Costo: 2087.33  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 11 | Esito: Successes | DistTot: 49.88m| DistX: 49.84m | Costo: 2419.46  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 12 | Esito: Successes | DistTot: 47.99m| DistX: 47.15m | Costo: 18477.75 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 13 | Esito: Successes | DistTot: 49.73m| DistX: 49.73m | Costo: 2168.91  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 14 | Esito: Successes | DistTot: 40.29m| DistX: 40.29m | Costo: 1435.45  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 15 | Esito: Successes | DistTot: 49.87m| DistX: 49.84m | Costo: 2397.94  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 16 | Esito: Successes | DistTot: 45.76m| DistX: 45.76m | Costo: 1840.72  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 17 | Esito: Successes | DistTot: 42.78m| DistX: 42.77m | Costo: 1653.90  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 18 | Esito: Successes | DistTot: 46.25m| DistX: 46.25m | Costo: 1879.96  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 19 | Esito: Successes | DistTot: 49.36m| DistX: 48.63m | Costo: 20889.01 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 20 | Esito: Successes | DistTot: 40.31m| DistX: 40.29m | Costo: 1484.02  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 21 | Esito: Successes | DistTot: 46.74m| DistX: 46.41m | Costo: 9788.95  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 22 | Esito: Successes | DistTot: 46.75m| DistX: 46.75m | Costo: 1919.59  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 23 | Esito: Successes | DistTot: 40.30m| DistX: 40.27m | Costo: 1672.52  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 24 | Esito: Successes | DistTot: 50.75m| DistX: 50.68m | Costo: 10329.14 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 25 | Esito: Successes | DistTot: 49.26m| DistX: 48.97m | Costo: 11646.20 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 26 | Esito: Successes | DistTot: 46.12m| DistX: 45.96m | Costo: 5638.45  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 27 | Esito: Successes | DistTot: 50.06m| DistX: 50.03m | Costo: 3186.69  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 28 | Esito: Successes | DistTot: 48.14m| DistX: 47.92m | Costo: 7403.07  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 29 | Esito: Successes | DistTot: 49.96m| DistX: 49.82m | Costo: 8541.39  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 30 | Esito: Successes | DistTot: 47.83m| DistX: 47.76m | Costo: 2191.73  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 31 | Esito: Successes | DistTot: 48.74m| DistX: 48.74m | Costo: 2082.09  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 32 | Esito: Successes | DistTot: 45.76m| DistX: 45.76m | Costo: 1840.72  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 33 | Esito: Successes | DistTot: 50.71m| DistX: 50.71m | Costo: 10908.94 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 34 | Esito: Successes | DistTot: 49.77m| DistX: 49.71m | Costo: 4560.59  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 35 | Esito: Successes | DistTot: 42.23m| DistX: 42.10m | Costo: 4548.15  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 36 | Esito: Successes | DistTot: 40.30m| DistX: 40.29m | Costo: 1479.84  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 37 | Esito: Successes | DistTot: 50.54m| DistX: 50.52m | Costo: 7960.96  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 38 | Esito: Successes | DistTot: 50.62m| DistX: 50.62m | Costo: 9303.57  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 39 | Esito: Successes | DistTot: 49.86m| DistX: 49.84m | Costo: 2409.65  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 40 | Esito: Successes | DistTot: 50.51m| DistX: 50.47m | Costo: 7290.45  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 41 | Esito: Successes | DistTot: 43.28m| DistX: 43.27m | Costo: 1704.11  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 42 | Esito: Successes | DistTot: 50.25m| DistX: 50.23m | Costo: 4598.68  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 43 | Esito: Successes | DistTot: 43.27m| DistX: 43.27m | Costo: 1653.46  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 44 | Esito: Successes | DistTot: 49.82m| DistX: 49.79m | Costo: 2296.93  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 45 | Esito: Successes | DistTot: 50.13m| DistX: 50.13m | Costo: 3723.13  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 46 | Esito: Successes | DistTot: 49.60m| DistX: 49.45m | Costo: 6478.65  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 47 | Esito: Successes | DistTot: 50.18m| DistX: 50.05m | Costo: 8105.22  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 48 | Esito: Successes | DistTot: 48.69m| DistX: 48.32m | Costo: 12873.74 | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 49 | Esito: Successes | DistTot: 49.91m| DistX: 49.73m | Costo: 7746.81  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# Target 50 | Esito: Successes | DistTot: 50.24m| DistX: 50.23m | Costo: 4593.78  | Step Stop: 2499 | Uscite dal box: 0   | Correlazione: Nessuna
# ======================================================================
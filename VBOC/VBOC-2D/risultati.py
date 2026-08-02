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
#N20: 10,2,1
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

#######################################



# RISULTATI TEST CON MPC-NN APF E MPC NAIVE
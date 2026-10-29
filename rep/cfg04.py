"""Celle del 04 tolte nella replica (indici di build_nb.parse)."""
S2 = set(range(20, 36)) - {25, 26}            # §2: esempio sintetico (si tengono le funzioni di CEBRA, §2.3)
S3 = set(range(39, 51)) - {40}                # §3.2-3.3: validazione dei passi (N_ITER = 1000) e mappa dei 10 000 passi
S6_ITER = {74, 75}                            # §6.2: passi di CEBRA-Behavior (N_ITER_BEH = 1000)
S6_ALL = set(range(72, 84))                   # §6 intero (solo la sessione di riferimento lo tiene)
S7_8 = set(range(84, 99))                     # §7 sensibilità e alternative, §8 riepilogo della parte A
S10 = set(range(113, 134))                    # §10: LFADS sui dati sintetici
S11 = set(range(136, 155)) - {138}            # §11.1-11.4 figure del primo allenamento, §11b griglia
S13B_15 = set(range(189, 236))                # §13.5-13.7 meno trial, §14 alternative, §15 riepilogo
def skip(ref):
    return S2 | S3 | S7_8 | S10 | S11 | S13B_15 | (S6_ITER if ref else S6_ALL)

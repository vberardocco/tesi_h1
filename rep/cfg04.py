"""Cells of 04 removed in the replication (indices of build_nb.parse)."""
S2 = set(range(20, 36)) - {25, 26}            # §2: synthetic example (the CEBRA functions of §2.3 are kept)
S3 = set(range(39, 51)) - {40}                # §3.2-3.3: validation of the steps (N_ITER = 1000) and the 10 000-step map
S6_ITER = {74, 75}                            # §6.2: CEBRA-Behavior steps (N_ITER_BEH = 1000)
S6_ALL = set(range(72, 84))                   # whole §6 (only the reference session keeps it)
S7_8 = set(range(84, 99))                     # §7 sensitivity and alternatives, §8 summary of part A
S10 = set(range(113, 134))                    # §10: LFADS on the synthetic data
S11 = set(range(136, 155)) - {138}            # §11.1-11.4 figures of the first training, §11b grid
S13B_15 = set(range(189, 236))                # §13.5-13.7 fewer trials, §14 alternatives, §15 summary
def skip(ref):
    return S2 | S3 | S7_8 | S10 | S11 | S13B_15 | (S6_ITER if ref else S6_ALL)

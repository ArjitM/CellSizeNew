import numpy as np
import pandas as pd
import os


loc = "/Users/arjitmisra/Documents/Kramer_Lab/RAW/RAWcopy/"

conditions = [
    # 'vit_A_free',
    # 'WT',
    # 'RD1-P2X7KO',
    # 'RD1',
    'VAF_new_cohort/contiguous',
    'VAF_new_cohort/reversal',
    'YFP-RA-viruses'
]

def getImageDirectories(locations):
    prefixes = []

    def recursiveDirectories(loc):
        nonlocal prefixes
        try:
            for d in next(os.walk(loc))[1]:
                if 'normal' in d or '_RFP' in d:
                    prefixes.append(loc + d + '/')
                    # print(loc + d + '/')
                else:
                    recursiveDirectories(loc + d + '/')
        except StopIteration:
            pass

    for loc in locations:
        recursiveDirectories(loc)
    return prefixes

for cond in conditions:
    dirs = getImageDirectories([loc + cond + "/"])
    somaSize = np.array([])
    nucleusSize = np.array([])
    for d in dirs:
        df = pd.read_excel(d + "areas.xlsx")
        areas = df["Areas"].to_numpy()
        if "rfp" in d or "ucleus" in d:
            nucleusSize = np.concatenate((nucleusSize, areas))
        else:
            somaSize = np.concatenate((somaSize, areas))

        s1 = pd.Series(somaSize, name="Soma")
        s2 = pd.Series(nucleusSize, name="Nucleus")
        outDf = pd.concat([s1, s2], axis=1)
        outDf.to_excel(loc + cond + ".xlsx")






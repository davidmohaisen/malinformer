import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_chord_diagram import chord_diagram

def manipulator(matrix):
    for i in range(len(matrix)):
        matrix[i][i] = 1-sum(matrix[i])
    return matrix

# names = ["Gafgyt", "Mirai", "Xorddos", "Tsunami", "Generica", "Ganiw", "Dofloo", "Setag", "Elknot", "Local"]
names = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]

# fn = "famTopKFeatsInFam2-orderedByFamiliesInPaper"
# fn = "top50PercFeatOverlapMatrix_final"
fn = "top20KFeatOverlapMatrix_final"
matrix = pickle.load(open("../data/"+fn+".pickle", "rb"))

print(matrix.shape)
matrix = list(matrix)
matrix = [list(itm) for itm in matrix]
# for i in range(len(matrix)):
#     matrix[i][i] = 0
matrix = manipulator(matrix)
flux = matrix
del matrix

print(flux)

grads = (True, False, False, True)                # gradient
gaps  = (0.01, 0, 0.02, 0)                        # gap value
sorts = ("size", "size", "distance", "distance")  # sort type
cclrs = (None, None, "slategrey", None)           # chord colors
nrota = (False, False, True, True)                # name rotation
cmaps = (None, None, None, "summer")              # colormap
fclrs = "black"                                    # fontcolors

for grd, gap, srt, cc, nr, cm in zip(grads, gaps, sorts, cclrs, nrota, cmaps):
    chord_diagram(flux, names, gap=gap, use_gradient=grd, sort=srt,
                  cmap=cm, chord_colors=cc, rotate_names=nr, fontcolor=fclrs)#, pad=5)

    str_grd = "_gradient" if grd else ""

    plt.savefig(
        "../data/imagesR1/{}{}_sort-{}.png".format(fn, str_grd, srt),
                dpi=600, transparent=True, bbox_inches='tight',
                pad_inches=0.02)

plt.show()

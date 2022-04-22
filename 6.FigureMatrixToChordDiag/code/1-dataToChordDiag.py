from chord import Chord
import holoviews as hv
import pickle

names = ["Gafgyt", "Mirai", "Xorddos", "Tsunami", "Generica", "Ganiw", "Dofloo", "Setag", "Elknot", "Local"]

matrix = pickle.load(open("../data/top20KFeatOverlapMatrix_final.pickle", "rb"))
print(matrix.shape)
matrix = list(matrix)
matrix = [list(itm) for itm in matrix]
for i in range(len(matrix)):
    matrix[i][i] = 0
print(type(matrix))
print(matrix)
# Chord(matrix, names, padding=0.3).to_html()
hv.Chord(matrix)
hv.extension("bokeh")
hv.extension("matplotlib")
hv.output(fig='xyz.pdf', size=250)
# hv.Chord(links)



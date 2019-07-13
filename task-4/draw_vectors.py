# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

vectors = pd.read_csv('vectors.csv')

types = vectors['Attribute']
x = vectors['V1']
y = vectors['V2']

fig, ax = plt.subplots()
ax.set_title('2-dimensional metadata embeddings', fontsize=18)
ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=12)
ax.scatter(x, y, color='r', alpha=0.7, marker=MarkerStyle(marker='x', fillstyle=None))

for i, txt in enumerate(types):
    ax.annotate(txt, (x[i], y[i]),fontsize=6)
    
# inset axes....
axins = ax.inset_axes([0.7, 0.6, 0.4, 0.5])
axins.scatter(x, y, color='r', marker=MarkerStyle(marker='x', fillstyle=None))
for i, txt in enumerate(types):
    axins.annotate(txt, (x[i], y[i]),fontsize=8)
# sub region of the original image
x1, x2, y1, y2 = -0.3, -0.1,-0.02, 0.23
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)

plt.show()
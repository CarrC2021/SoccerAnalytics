import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import os
from xGmodel import xGmodel
from WyscoutWrapper import WyscoutWrapper


HEADER = 403
CWD = os.path.dirname(os.path.realpath(__file__))
filename = f"{CWD}/models/headers.pickle"

model = xGmodel(filename)

# save the model summary as latex table in a file
with open(f"{CWD}/models/headers_model_summary.tex", "w") as f:
    f.write(model.model.summary().as_latex())

num = 68
# Create a 2D map of xG
pgoal_2d = np.zeros((num, num))
for x in range(num):
    for y in range(num):
        pgoal_2d[x, y] = model.calc_xG(x, y)

# plot pitch
pitch = VerticalPitch(
    line_color="black",
    half=True,
    pitch_type="custom",
    pitch_length=105,
    pitch_width=68,
    line_zorder=2,
)
fig, ax = pitch.draw()
# plot probability
pos = ax.imshow(
    pgoal_2d,
    extent=[-1, 68, 68, -1],
    aspect="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
    zorder=1,
)
fig.colorbar(pos, ax=ax)
# make legend
ax.set_title("Probability of Header Scoring", fontsize=20)
plt.xlim((0, 68))
plt.ylim((0, 60))
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
# save the plot
plt.savefig(f"plots/headers_xG.png")

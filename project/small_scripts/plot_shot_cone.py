import mplsoccer
import sql_wrapper

sql = sql_wrapper.SQLWrapper()

shots = sql.get_shots_and_passes(2499719)

shots = shots[shots["subEventId"] == 100]

shot = shots.iloc[3]

shot["X"] = 105 - shot["X"]
shot["end_x"] = 105 - shot["end_x"]

pitch = mplsoccer.VerticalPitch(
    pitch_type="custom", pitch_length=105, pitch_width=68, half=True, pad_bottom=-20
)

fig, ax = pitch.draw(figsize=(16, 12))

# We will use mplsoccer's grid function to plot a pitch with a title axis.
fig, axs = pitch.grid(
    figheight=8,
    endnote_height=0,  # no endnote
    title_height=0.1,
    title_space=0.02,
    # Turn off the endnote/title axis. I usually do this after
    # I am happy with the chart layout and text placement
    axis=False,
    grid_height=0.83,
)

line = pitch.lines(
    shot["X"],
    shot["Y"],
    shot["end_x"],
    shot["end_y"],
    comet=True,
    label="shot",
    color="#cb5a4c",
    ax=axs["pitch"],
)

# scatter the circle where the shot was taken from
pitch.scatter(shot["X"], shot["Y"], ax=axs["pitch"], s=100, color="#cb5a4c", zorder=1.1)

# plot the angle to the goal
pitch.goal_angle(
    shot["X"],
    shot["Y"],
    ax=axs["pitch"],
    alpha=0.2,
    zorder=1.1,
    color="#cb5a4c",
    goal="right",
)

# plot this and save
fig.savefig("plots/shot_cone.png", dpi=300, bbox_inches="tight")

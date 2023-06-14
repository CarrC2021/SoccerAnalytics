from mplsoccer import Radar, FontManager, grid
import pandas as pd
import matplotlib.pyplot as plt


# Creating a class to make a radar chart using mplsoccer
class RadarWrapper:
    def __init__(
        self,
        data: list,
        column_names: list,
        lows: list,
        highs: list,
        lower_is_better: list,
        plot_percentiles: bool = False,
        is_comparison: bool = False,
    ):
        self.data = data
        if is_comparison:
            self.data1 = data[0]
            self.data2 = data[1]
        self.column_names = column_names
        self.lows = lows
        self.lower_is_better = lower_is_better
        self.highs = highs
        self.radar = Radar(
            self.column_names,
            self.lows,
            self.highs,
            lower_is_better=lower_is_better,
            # whether to round any of the labels to integers instead of decimal places
            round_int=[False] * len(self.column_names),
            num_rings=4,  # the number of concentric circles (excluding center circle)
            # if the ring_width is more than the center_circle_radius then
            # the center circle radius will be wider than the width of the concentric circles
            ring_width=1,
            center_circle_radius=1,
        )
        URL1 = (
            "https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/"
            "SourceSerifPro-Regular.ttf"
        )
        self.serif_regular = FontManager(URL1)
        URL2 = (
            "https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/"
            "SourceSerifPro-ExtraLight.ttf"
        )
        self.serif_extra_light = FontManager(URL2)
        URL3 = (
            "https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/"
            "RubikMonoOne-Regular.ttf"
        )
        self.rubik_regular = FontManager(URL3)
        URL4 = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf"
        self.robotto_thin = FontManager(URL4)
        URL5 = (
            "https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/"
            "RobotoSlab%5Bwght%5D.ttf"
        )
        self.robotto_bold = FontManager(URL5)
        self.percentiles = plot_percentiles

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def set_lows_highs(self, lows: list, highs: list) -> None:
        self.lows = lows
        self.highs = highs

    def plot_radar(self, player: str, team: str) -> None:
        fig, axs = grid(
            figheight=14,
            grid_height=0.915,
            title_height=0.06,
            endnote_height=0.025,
            title_space=0,
            endnote_space=0,
            grid_key="radar",
            axis=False,
        )  # format axis as a radar
        self.radar.setup_axis(ax=axs["radar"])
        rings_inner = self.radar.draw_circles(
            ax=axs["radar"], facecolor="#ffb2b2", edgecolor="#fc5f5f"
        )
        radar_output = self.radar.draw_radar(
            self.data,
            ax=axs["radar"],
            kwargs_radar={"facecolor": "#aa65b2"},
            kwargs_rings={"facecolor": "#66d8ba"},
        )
        radar_poly, rings_outer, vertices = radar_output
        range_labels = self.radar.draw_range_labels(
            ax=axs["radar"], fontsize=15, fontproperties=self.robotto_thin.prop
        )
        param_labels = self.radar.draw_param_labels(
            ax=axs["radar"], fontsize=18, fontproperties=self.robotto_bold.prop
        )
        endnote_text = axs["endnote"].text(
            0.99,
            0.5,
            "Inspired By: StatsBomb / Rami Moghadam",
            fontsize=15,
            fontproperties=self.robotto_thin.prop,
            ha="right",
            va="center",
        )
        if self.percentiles:
            ednnte_text2 = axs["endnote"].text(
                0.01,
                0.5,
                "Percentile for stat in each category relative to same role",
                fontsize=15,
                fontproperties=self.robotto_bold.prop,
                ha="left",
                va="center",
            )
        title1_text = axs["title"].text(
            0.01,
            0.65,
            f"{player}",
            fontsize=25,
            fontproperties=self.robotto_bold.prop,
            ha="left",
            va="center",
        )
        title2_text = axs["title"].text(
            0.01,
            0.25,
            f"{team}",
            fontsize=20,
            fontproperties=self.robotto_thin.prop,
            ha="left",
            va="center",
            color="#B6282F",
        )
        # save the radar
        fig.savefig(
            f"plots/{player}_radar.png", dpi=300, bbox_inches="tight", pad_inches=0.2
        )
        # clear the radar and remove the plot from memory
        fig.clf()
        plt.close(fig)

    def plot_comparison_radar(self, player1: str, player2: str) -> None:
        fig, axs = grid(
            figheight=14,
            grid_height=0.915,
            title_height=0.06,
            endnote_height=0.025,
            title_space=0,
            endnote_space=0,
            grid_key="radar",
            axis=False,
        )  # format axis as a radar
        self.radar.setup_axis(ax=axs["radar"])
        rings_inner = self.radar.draw_circles(
            ax=axs["radar"], facecolor="#ffb2b2", edgecolor="#fc5f5f"
        )
        radar_output = self.radar.draw_radar_compare(
            self.data1,
            self.data2,
            ax=axs["radar"],
            kwargs_radar={"facecolor": "#00f2c1", "alpha": 0.5},
            kwargs_compare={"facecolor": "#d80499", "alpha": 0.5},
        )
        radar_poly, rings_outer, vertices, vertices2 = radar_output
        range_labels = self.radar.draw_range_labels(
            ax=axs["radar"], fontsize=15, fontproperties=self.robotto_thin.prop
        )
        param_labels = self.radar.draw_param_labels(
            ax=axs["radar"], fontsize=18, fontproperties=self.robotto_bold.prop
        )
        title3_text = axs["title"].text(
            0.5,
            0.3,
            "Stats Per 90 (Possession Adjusted)",
            fontsize=25,
            fontproperties=self.robotto_bold.prop,
            ha="center",
            va="center",
        )
        endnote_text = axs["endnote"].text(
            0.99,
            0.5,
            "Inspired By: StatsBomb / Rami Moghadam",
            fontsize=15,
            fontproperties=self.robotto_thin.prop,
            ha="right",
            va="center",
        )
        title1_text = axs["title"].text(
            0.01,
            0.65,
            f"{player1}",
            fontsize=25,
            fontproperties=self.robotto_bold.prop,
            ha="left",
            va="center",
            color="#00f2c1",
        )
        title2_text = axs["title"].text(
            0.01,
            0.25,
            f"{player2}",
            fontsize=20,
            fontproperties=self.robotto_thin.prop,
            ha="left",
            va="center",
            color="#d80499",
        )
        if self.percentiles:
            ednnte_text2 = axs["endnote"].text(
                0.01,
                0.5,
                "Percentile for stat in each category relative to same role",
                fontsize=15,
                fontproperties=self.robotto_bold.prop,
                ha="left",
                va="center",
            )
            fig.savefig(
                f"plots/{player1} vs {player2}_radar_percentiles.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.2,
            )
        else:
            fig.savefig(
                f"plots/{player1} vs {player2}_radar.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.2,
            )
        fig.clf()
        plt.close(fig)

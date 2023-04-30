from mplsoccer import Radar, FontManager, grid
import pandas as pd

# Creating a class to make a radar chart using mplsoccer
class RadarWrapper():
    def __init__(self, data: list, column_names: list, lows: list, highs: list, lower_is_better: list):
        self.data = data
        self.column_names = column_names
        self.lows = lows
        self.lower_is_better = lower_is_better
        self.highs = highs
        self.radar = Radar(self.column_names, self.lows, self.highs,
              lower_is_better=lower_is_better,
              # whether to round any of the labels to integers instead of decimal places
              round_int=[False]*len(self.column_names),
              num_rings=4,  # the number of concentric circles (excluding center circle)
              # if the ring_width is more than the center_circle_radius then
              # the center circle radius will be wider than the width of the concentric circles
              ring_width=1, center_circle_radius=1)
        URL1 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-Regular.ttf')
        self.serif_regular = FontManager(URL1)
        URL2 = ('https://raw.githubusercontent.com/googlefonts/SourceSerifProGFVersion/main/fonts/'
        'SourceSerifPro-ExtraLight.ttf')
        self.serif_extra_light = FontManager(URL2)
        URL3 = ('https://raw.githubusercontent.com/google/fonts/main/ofl/rubikmonoone/'
        'RubikMonoOne-Regular.ttf')
        self.rubik_regular = FontManager(URL3)
        URL4 = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Thin.ttf'
        self.robotto_thin = FontManager(URL4)
        URL5 = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
        self.robotto_bold = FontManager(URL5)

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def set_lows_highs(self, lows: list, highs: list) -> None:
        self.lows = lows
        self.highs = highs

    def plot_radar(self, player: str, team: str) -> None:
        fig, axs = grid(figheight=14, grid_height=0.915, title_height=0.06, endnote_height=0.025,
                title_space=0, endnote_space=0, grid_key='radar', axis=False)  # format axis as a radar
        self.radar.setup_axis(ax=axs['radar'])
        rings_inner = self.radar.draw_circles(ax=axs['radar'], facecolor='#ffb2b2', edgecolor='#fc5f5f')
        radar_output = self.radar.draw_radar(self.data, ax=axs['radar'],
                                kwargs_radar={'facecolor': '#aa65b2'},
                                kwargs_rings={'facecolor': '#66d8ba'})
        radar_poly, rings_outer, vertices = radar_output
        range_labels = self.radar.draw_range_labels(ax=axs['radar'], fontsize=25,
                                       fontproperties=self.robotto_thin.prop)
        param_labels = self.radar.draw_param_labels(ax=axs['radar'], fontsize=25,
                                       fontproperties=self.robotto_thin.prop)

        title1_text = axs['title'].text(0.01, 0.65, f'{player}', fontsize=25,
                                fontproperties=self.robotto_bold.prop, ha='left', va='center')
        title2_text = axs['title'].text(0.01, 0.25, f'{team}', fontsize=20,
                                fontproperties=self.robotto_thin.prop,
                                ha='left', va='center', color='#B6282F')
        # save the radar
        fig.savefig(f'{player}_radar.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        

    
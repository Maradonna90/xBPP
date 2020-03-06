import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k", visible=True)
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                x_grid = self.xaxis.get_gridlines()
                for gl in x_grid:
                    gl.set_visible(False)
                #x_grid.set_visible(False)
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars), alpha=0.5)
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    data = [
        #TODO: RETRIEVE THE STAAATS
        ['xBPP', 'WHIP', 'ERA-', 'xFIP', 'SIERA', 'Pitches', 'fWAR'],
        # playerdata
        ('Pitching', [
            [0.191, 1.52, 148, 4.87, 4.93, 1819, 1.3],
            [0.184, 1.13, 89, 2.68, 2.93, 758, 0.7],
            [0.203, 1.19, 86, 4.61, 4.47, 2402, 2.1]]),
            #[0.191, 1.52, 4.8, 4.87, 4.93, 5, 1.3],
            #[0.184, 1.13, 0.89, 2.68, 2.93, 3.8, 0.7],
            #[0.203, 1.19, 0.86, 4.61, 4.47, 2.7, 2.1]]),

            ]
    return data

if __name__ == '__main__':
    #FIXME: use proper color pallette
    # colors = ['b', 'r', 'g', 'm', 'y']
    # lower, lower, lower, lower, lower, HIGHER, HIGHER
    dir_vals = ["L", "L", "L", "L", "L", "H", "H"]
    max_vals = [0.207191, 1.52, 119, 5.27, 5.28, 3490, 7.4]
    min_vals = [0.165177, 0.80, 55, 2.48, 2.62, 402, 0.5]
    #LOW: max-n/max-min
    #HIGH: (max-min)-(max-n)/max-min
    data = example_data()
    N = len(data[0])
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)
    title, case_data = data[0]
    scaled_data = []
    for d in case_data:
        scaled = []
        for i, (max, min, type) in enumerate(zip(max_vals, min_vals, dir_vals)):
            if type == "L":
                scaled_val = (max-d[i])/(max-min)
                scaled.append(scaled_val)
            elif type == "H":
                scaled_val = ((max-min)-(max-d[i]))/(max-min)
                scaled.append(scaled_val)
        scaled_data.append(scaled)


    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    rgrid = [0, 0.25, 0.5, 0.75, 1]
    ax.set_rgrids(rgrid, visible=False)
    ax.set_rlabel_position(0)
    for i, t in enumerate(theta):
        for r in rgrid:
            val_range = max_vals[i] - min_vals[i]
            if dir_vals[i] == 'L':
                label = max_vals[i]-r*val_range
            elif dir_vals[i] == 'H':
                label = min_vals[i]+r*val_range
            ax.text(t, r, "{:.3g}".format(label))
    ax.set_title(title,  position=(0.5, 1.1), ha='center')
    for d in scaled_data:
        line = ax.plot(theta, d)
        ax.fill(theta, d,  alpha=0.25)
    ax.set_varlabels(spoke_labels)
    labels = ('Jordan Zimmermann', 'Austin Adams', 'Chris Bassitt')
    legend = plt.legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')


    plt.savefig("pitcher_radar_chart.pdf")

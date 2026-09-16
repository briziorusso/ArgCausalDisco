import unittest
from unittest.mock import patch
import pandas as pd
import plotly.graph_objects as go
from utils.plotting import double_bar_chart_plotly


class SIDPlotTests(unittest.TestCase):
    def plot(self, **kwargs):
        frame = pd.DataFrame({'dataset': ['Child'], 'model': ['Random'],
                              'p_SID_low_mean': [8.47], 'p_SID_low_std': [1.],
                              'p_SID_high_mean': [9.81], 'p_SID_high_std': [2.]})
        figures = []
        with patch.object(go.Figure, 'show', lambda fig: figures.append(fig)):
            double_bar_chart_plotly(frame, ['p_SID_low', 'p_SID_high'],
                                   {'random': 'Random'}, {'random': '#888888'}, ['Random'], **kwargs)
        return figures[0]

    def test_bounds_share_axis_and_error_bars_fit(self):
        fig = self.plot()
        self.assertEqual(fig.layout.yaxis.range, fig.layout.yaxis2.range)
        self.assertEqual(fig.layout.yaxis2.matches, 'y')
        self.assertGreater(fig.layout.yaxis.range[1], 11.81)

    def test_conflicting_explicit_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            self.plot(range_y1=[0, 10], range_y2=[0, 16])


if __name__=='__main__':unittest.main()

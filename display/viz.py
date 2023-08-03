# from bqplot import (
#     Axis,
#     Figure,
#     Lines,
#     LinearScale,
#     OrdinalColorScale,
#     Scatter
# )

import bqplot.pyplot as plt

from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import color

BACKGROUND_COLOR = "#1a1a1a"

def gen_single_data_plot(
    frame,
    x_label = "dates",
    y_label = "daily change (%)",
    title = ''
    
) -> widgets.VBox:
    frame = frame.copy()
    
    tooltip = widgets.VBox(
        [],
        layout={
            'width': '335px',
            'height': '195px',
            'padding': '10px',
        }
    )
    
    return widgets.VBox(
        [fig],
        layout = {'width': '99%'}
    )

def _gen_legend(
    values,
    colors,
    fontsize='10px',
    sep='space'
) -> widgets.HTML:
    """
    Bqplot isn't very good at creating legends so we'll manually create our
    own legend in HTML.

    Parameters
    ----------
    values : list of str
    colors : list of str
    fontsize : str
        Fontsize of the legend text and icons.
    sep : {'space', 'line'}, default 'space'
        Joins legend patches with a long space if 'space', joins with a newline
        if 'line'. The latter ensures that there is only one patch per line.

    Returns
    -------
    Legend widget.
    """
    patches = []
    styles = []

    # We need to generate legend specific patch IDs in case a single app has
    # multiple legends.
    legend_id = randint(1, 100000)

    # JLabs doesn't color HTML symbols so use style classes to make sure
    # ticker and symbol are both the correct color.
    style_format = """
    .patch{i} {{
        color: transparent;
        display: inline;
        font-size: {fontsize};
        text-shadow: 0 0 0 {color};
    }}
    """
    patch_format = '<span class="{style_class}">&#9670;&nbsp;{ticker}</span>'

    for i, (val, col) in enumerate(zip(values, colors)):
        patch_id = i + legend_id
        style = style_format.format(
            i=patch_id,
            fontsize=fontsize,
            color=col
        )
        styles.append(style)
        patch = patch_format.format(
            style_class=f'patch{patch_id}',
            ticker=val
        )
        patches.append(patch)

    if sep is 'space':
        patch_sep = '&emsp;'
    elif sep is 'line':
        patch_sep = '<br>'

    patches_string = patch_sep.join(patches)
    class_styles = ''.join(styles)

    html = f"""
    <style>
        {class_styles}
    </style>
    <p style="line-height:145%">
        {patches_string}
    </p>
    """
    return widgets.HTML(html)

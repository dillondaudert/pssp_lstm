from matplotlib import pyplot as plt
import matplotlib
import plotly.graph_objs as go
import numpy as np

def _plot_line_CI(ax,
                  x,
                  y,
                  sorted_x,
                  low_CI,
                  high_CI,
                  color,
                  label
                 ):
    """
    Plot a single series and its 95% CI in axes.
    """
    ax.plot(x, y, lw=1, color=color, alpha=1, label=label)
    # shade the CI
    ax.fill_between(sorted_x, 
                    low_CI, 
                    high_CI, 
                    color=color, 
                    alpha=0.4, 
                   )

def line_CI_plot(xs, 
                 ys, 
                 sorted_xs, 
                 low_CIs, 
                 high_CIs, 
                 labels,
                 colors,
                 x_label, 
                 y_label, 
                 title):
    # plot size to 14" x 7"
    matplotlib.rc("figure", figsize=(14, 7))
    # font size to 14
    matplotlib.rc("font", size=14)
    
    # create plot object
    fig, ax = plt.subplots()
    
    ax.set_yscale("log", nonposy="clip")
    ax.grid(True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_locator(plt.MultipleLocator(5000))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2500))
    
    # plot data, set linewidth, color, transparency,
    # provide label for the legend
    for i in range(len(xs)):
        _plot_line_CI(ax, xs[i], ys[i], sorted_xs[i], low_CIs[i], high_CIs[i], colors[i], labels[i])
    
    # label axes and provide title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # display legend
    ax.legend(loc="best")
    fig
    
    
def line_CI_plot_2(ys, 
                   low_CIs, 
                   high_CIs, 
                   labels,
                   colors):
    
    data = []
    for i in range(len(labels)):
        # add scatter plots for lower, mean, upper bounds
        # for each series
        lower_bound = go.Scatter(
            name=labels[i]+" low CI",
            x=ys[i].index,
            y=low_CIs[i],
            mode="lines",
            marker=dict(color=colors[i], opacity=.4),
            line=dict(width=0),
            showlegend=False,)
        trace = go.Scatter(
            name=labels[i],
            x=ys[i].index,
            y=ys[i],
            mode="lines",
            line=dict(color=colors[i]),
            fill="tonexty")
        upper_bound = go.Scatter(
            name=labels[i]+" high CI",
            x=ys[i].index,
            y=high_CIs[i],
            mode="lines",
            marker=dict(color=colors[i]),
            line=dict(width=0),
            fill="tonexty",
            showlegend=False)

        data += [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title="Cross Entropy", type="log", autorange=True),
        title="Validation Loss",
        showlegend=True)
    fig = go.FigureWidget(data=data, layout=layout)
    return fig

def plot_confusion_matrix(title,
                          cm_values,
                          classes,
                          normalize=True,
                         ):
    if normalize:
        totals = cm_values.sum(axis=0)
        cm_values = np.divide(cm_values, totals, out=np.zeros_like(cm_values), where=totals!=0.)
        
    trace = {
        "x": classes,
        "y": classes[::-1],
        "z": cm_values[::-1, :],
        "colorscale": "Jet",
        "type": "heatmap"
    }
    data = go.Data([trace])
    layout = {
        "barmode": "overlay",
        "title": title,
        "xaxis": {
            "title": "Predicted Label",
            "titlefont": {
                "family": "Courier New, monospace",
                "size": 18
            }
        },
        "yaxis": {
            "title": "True Label",
            "titlefont": {
                "family": "Courier New, monospace",
                "size": 18
            }
        }
    }
    fig = go.FigureWidget(data=data, layout=layout)
    return fig
    
    
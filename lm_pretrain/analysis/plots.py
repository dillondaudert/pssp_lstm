from matplotlib import pyplot as plt
import matplotlib

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
    # remove top and right frame lines
    #matplotlib.rc("axes.spines", top=False, right=False)
    # remove grid lines
    matplotlib.rc("axes", grid=False)
    # set background color
    matplotlib.rc("axes", facecolor="white")
    
    # create plot object
    _, ax = plt.subplots()
    
    for i in range(len(xs)):
        _plot_line_CI(ax, xs[i], ys[i], sorted_xs[i], low_CIs[i], high_CIs[i], colors[i], labels[i])
    
    # plot data, set linewidth, color, transparency,
    # provide label for the legend
    # label axes and provide title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # display legend
    ax.legend(loc="best")
    
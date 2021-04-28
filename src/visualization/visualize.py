import matplotlib.pyplot as plt

def save_plot(plot, path, filename):
    """Save plot as png to desired path.

    Args:
        path (string): Path to where you would like to save plot
        filename (string): Name for plot, be sure to end Name with appropriate image type (ie. .png or .jpeg)
    """
    if filename.endswith('.png'):
        plot.figure.savefig(path + filename, bbox_inches='tight')
        print('Successfully saved image to path');
    else:
        raise Exception('Failed to include appropriate ending to filename')
from pga_analysis.golf_stats import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)

def add_scraped_data(pga_data):
    """
    Add scraped data of tournament winnings & driving distances to main dataset.
    """
    winnings_and_driving = pd.read_csv(r'data/winnings_and_driving_distances_by_player.csv')
    pga_data = pga_data.merge(winnings_and_driving.drop(['Tournament', 'year'], axis=1),
                              how='left', on=['player', 'tournament'])

    pga_data_added = pga_data.dropna(subset=['winnings', 'avg_driving_distance'])
    pga_data_added['winnings'] = cast_winnings_to_float(pga_data_added)

    return pga_data_added


def plot_bryson_drive():
    """
    Presents images of Bryson's tee shot and second shot on Hole 6 @ Arnold Palmer Invitational.
    """
    bryson_shot_1 = mpimg.imread(r'images/bryson_shot_1.png')
    bryson_shot_2 = mpimg.imread(r'images/bryson_shot_2.png')

    fig, ax = plt.subplots(1, 2, figsize=[30, 20])
    ax[0].imshow(bryson_shot_1)
    ax[0].axis('off')
    ax[1].imshow(bryson_shot_2)
    ax[1].axis('off')

def tiger_woods_____only_just():
    """
    Show Tiger doing fingers pose.
    """
    only_just = mpimg.imread(r'images/only_just.png')
    fig, ax = plt.subplots(1, 1, figsize=[7.5,5])
    ax.imshow(only_just)
    ax.axis('off')

def set_size(width, height):
    """
    Helper function to set size of Seaborn plots.
    """
    ax = plt.gca()
    fig_width = float(width) / (ax.figure.subplotpars.right - ax.figure.subplotpars.left)
    fig_height = float(height) / (ax.figure.subplotpars.top - ax.figure.subplotpars.bottom)
    ax.figure.set_size_inches(fig_width, fig_height)
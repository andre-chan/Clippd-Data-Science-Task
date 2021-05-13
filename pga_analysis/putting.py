"""
Analysis of Putting in the 2020-21 PGA Tour Season.
"""
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from pga_analysis.golf_stats import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings

warnings.simplefilter("ignore", UserWarning)

def train_strokes_to_hole_out_model(putt_distance_to_pin, strokes_to_hole_out, model_type, radius=0.5):
    """
    Train local regression models that predict the number of strokes to hole out against DTP.
    """
    if model_type == 'knn':
        strokes_model = KNeighborsRegressor(n_neighbors=100, weights='uniform')
    elif model_type == 'radius':
        strokes_model = RadiusNeighborsRegressor(radius=radius, weights='uniform')
    strokes_model.fit(np.array(putt_distance_to_pin).reshape(-1, 1), strokes_to_hole_out)

    return strokes_model


def pga_tour_website_putts_by_yardage():
    """
    Returns expected putts from 0-20 feet, from official PGA Tour statistics.
    """
    return [1.000, 1.001, 1.009, 1.053, 1.147, 1.256, 1.357, 1.443, 1.515, 1.575, 1.626, 1.669, 1.705, 1.737, 1.765,
            1.790, 1.811, 1.83, 1.848, 1.863, 1.878]


def compute_putting_strokes_gained(putting_data, first_shot_on_green, by_player=False, aggregation='mean'):
    """
    (This function is a little complex - just to abstract code from the Jupyter notebook..)

    Compute putts gained for each hole from the first shot on the green.
    Putts gained is the difference between the predicted putts to hole out and the actual putts that were taken.

    Predictions the radius neighbours model by default, but if no neighbours exist in the radius, it
    falls back to the KNN model.

    by_player (bool): Returns {aggregation} SG by player.
    """
    x_train = putting_data['shot_start_distance_feet'].sample(100000, random_state=72)
    y_train = putting_data['strokes_from_here'].sample(100000, random_state=72)

    strokes_to_hole_out_model_knn = train_strokes_to_hole_out_model(x_train, y_train, 'knn')
    strokes_to_hole_out_model_radius = train_strokes_to_hole_out_model(x_train, y_train, 'radius')

    x_predict = np.array(first_shot_on_green['shot_start_distance_feet']).reshape(-1, 1)
    strokes_predictions_knn = strokes_to_hole_out_model_knn.predict(x_predict)
    strokes_predictions_radius = strokes_to_hole_out_model_radius.predict(x_predict)

    predicted_strokes = \
        pd.Series([strokes[0] if abs(strokes[0]) < 10 else strokes[1]
         for strokes in zip(strokes_predictions_radius, strokes_predictions_knn)])

    strokes_gained = \
        predicted_strokes.subtract(first_shot_on_green['strokes_from_here'].reset_index(drop=True))

    strokes_gained.index = first_shot_on_green.index

    if by_player:
        players = first_shot_on_green.player
        strokes_gained_by_player = pd.DataFrame({'player': players, 'strokes_gained': strokes_gained})
        if aggregation == 'mean':
            agg_strokes_gained_by_player = strokes_gained_by_player.groupby('player').mean()
        if aggregation == 'sum':
            agg_strokes_gained_by_player = strokes_gained_by_player.groupby('player').sum()
        return agg_strokes_gained_by_player


    return strokes_gained

def prepare_putting_data(pga_data):
    """
    Prepare putting data for analysis.
    """
    putting_data = pga_data[pga_data['shot_start_location'] == 'green']
    putting_data['total_putts'] = compute_total_putts(putting_data)
    putting_data['shot_start_distance_feet'] = putting_data['shot_start_distance_yards'] * 3

    first_shot_on_green = putting_data.loc[putting_data.groupby(hole_identifier).shot_number.idxmin()]

    return putting_data, first_shot_on_green


def plot_number_of_putts_by_yardage(putting_data, strokes_model=None):
    """
    For putts between 0-20 feet, plot predicted number of putts to hole-out.
    """
    if strokes_model is None:
        strokes_model = train_strokes_to_hole_out_model(
            putting_data['shot_start_distance_feet'], putting_data['strokes_from_here'], 'radius')

    putts_yardages = np.arange(0, 21).reshape(-1, 1)
    predicted_strokes_to_hole_out_by_yardage = strokes_model.predict(putts_yardages)

    fig, ax = plt.subplots(figsize=[13,3])
    plt.scatter(putts_yardages, predicted_strokes_to_hole_out_by_yardage)
    plt.plot(np.arange(0, 21), pga_tour_website_putts_by_yardage())

    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_xticklabels(np.arange(0, 21, 2))
    plt.xlabel('Distance to Pin (Feet)')
    plt.ylabel('# Putts')
    plt.ylim([1,2])
    plt.title('Dots = Predictions  /  Line = Official PGA Tour Stats')
    plt.show()


def plot_tommy_gainey_putts_curve(first_shot_to_green, putting_data):
    """
    Plot putts curve for Tommy Gainey, compared to PGA Tour average.
    """
    putting_tommy_gainey = first_shot_to_green[first_shot_to_green.player == 'Tommy Gainey']
    putts_dtp = np.arange(0, 31).reshape(-1, 1)

    strokes_model_tommy = train_strokes_to_hole_out_model(putting_tommy_gainey.shot_start_distance_feet,
                                                          putting_tommy_gainey.strokes_from_here, 'radius', radius=2)
    strokes_smoothed_tommy = pd.DataFrame({'putt_dtp': np.arange(0, 31), 'number_of_putts': strokes_model_tommy.predict(
        putts_dtp)})
    strokes_model_pga = train_strokes_to_hole_out_model(
        putting_data['shot_start_distance_feet'], putting_data['strokes_from_here'], 'radius')

    fig, ax = plt.subplots(figsize=[7,5])
    sns.lineplot(np.arange(0, 31), strokes_model_pga.predict(putts_dtp), ax=ax)
    sns.regplot('putt_dtp', 'number_of_putts', data=strokes_smoothed_tommy, lowess=True, ax=ax, scatter=False)

    red_patch = mpatches.Patch(color='palegoldenrod', label='Tommy Gainey')
    blue_patch = mpatches.Patch(color='cyan', label='PGA Tour Average')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlim([0, 30])
    plt.ylabel('# Putts')
    plt.xlabel('Distance to Pin (feet)')
    plt.title('Tommy Gainey: Bad SHORT Putter, Good LONG Putter')
"""
Analysis of Par 5 Tee shots in the 2020-21 PGA Tour Season.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from pga_analysis.helpers import set_size
plt.style.use('dark_background')


def common_shot_end_locations(pga_data: pd.DataFrame):
    """
    Returns shot_end_location only with over 100 counts in the data, to simplify analysis.
    """
    shot_end_location_counts = pga_data['shot_end_location_simplified'].value_counts()
    shot_end_location_more_than_100_counts = list(shot_end_location_counts[shot_end_location_counts >= 100].index)

    return shot_end_location_more_than_100_counts

def get_par_5_analysis_data(pga_data, shot_end_locations=None):
    """
    Prepares data for par-5 tee shot analysis.
    """
    if shot_end_locations is None:
        shot_end_locations = common_shot_end_locations(pga_data)
    par_5_tee_shots = [(pga_data['shot_number'] == 1) & (pga_data['hole_par'] == 5) &
                       pga_data['shot_end_location_simplified'].isin(shot_end_locations)][0]
    tee_shot_features = ['player', 'shot_end_distance_yards', 'shot_end_location_simplified', 'strokes_from_here']

    par_5_tee_shot_features = pga_data[par_5_tee_shots][tee_shot_features]
    par_5_tee_shot_features = pd.get_dummies(data=par_5_tee_shot_features, drop_first=True,
                                             columns=['shot_end_location_simplified'])

    par_5_tee_shot_features['strokes_to_finish'] = pga_data[par_5_tee_shots]['strokes_from_here'] - 1

    return par_5_tee_shot_features


def plot_distance_to_pin_distribution(par_5_tee_shot_features):
    """
    Plot the distribution of distance to the pin after the Tee Shot on Par 5s.
    """
    fig, ax = plt.subplots(figsize=[10, 2])
    ax = sns.kdeplot(x=par_5_tee_shot_features.shot_end_distance_yards, shade=True)
    plt.xlabel('Distance to Pin (Yards)')
    plt.ylabel(' ')
    plt.title('Par 5: Distance to Pin after the Tee Shot')
    plt.xlim([150, 350])
    ax.yaxis.set_ticks([])
    plt.tight_layout()
    plt.show()
    return ax


def plot_strokes_from_par_5_tee_shot(par_5_tee_shot_features):
    """
    Plot the predicted strokes to hole out against distance to pin on Par 5s.
    """
    # Train local regression models (KNN and Radius Neighbours)
    strokes_model_knn = train_strokes_to_finish_model(par_5_tee_shot_features, model_type='knn')
    strokes_model_radius = train_strokes_to_finish_model(par_5_tee_shot_features, model_type='radius')
    # Predict # strokes to hole-out for each yards, and plot.
    predicted_strokes_to_finish_by_yardage = predict_strokes_to_finish_by_yardage(strokes_model_knn,
                                                                                  strokes_model_radius)

    sns.lmplot(x='Distance to Pin (yards)', y='Pred. Strokes Left',
               data=predicted_strokes_to_finish_by_yardage, lowess=True, scatter=False)

    set_size(8.5, 2)
    plt.title('Par 5: Predicted Strokes to Hole-Out vs Distance to Pin')
    plt.tight_layout()
    plt.show()


def train_strokes_to_finish_model(par_5_tee_shot_features, model_type):
    """
    Train local regression models that predict the number of strokes to hole out against DTP
    """
    if model_type == 'knn':
        strokes_model = KNeighborsRegressor(n_neighbors=10, weights='uniform')
    elif model_type == 'radius':
        strokes_model = RadiusNeighborsRegressor(radius=5, weights='uniform')

    distance_to_pin = np.array(par_5_tee_shot_features.shot_end_distance_yards).reshape(-1, 1)
    strokes_to_finish = np.array(par_5_tee_shot_features.strokes_from_here).reshape(-1, 1) - 1

    strokes_model.fit(distance_to_pin, strokes_to_finish)

    return strokes_model


def predict_strokes_to_finish_by_yardage(strokes_model_knn, strokes_model_radius):
    """
    For yardages between 150 and 350, predict the number of strokes to hole out.
    This uses the radius neighbours model by default, but if no neighbours exist in the radius, it
    falls back to the KNN model.
    """
    yardage = np.arange(150, 351, 1).reshape(-1, 1)
    strokes_predictions_knn = strokes_model_knn.predict(yardage)
    strokes_predictions_radius = strokes_model_radius.predict(yardage)

    predicted_strokes = \
        [strokes[0] if abs(strokes[0]) < 10 else strokes[1]
         for strokes in zip(strokes_predictions_knn, strokes_predictions_radius)]
    predicted_strokes = np.concatenate(predicted_strokes, axis=0).reshape(-1, 1)
    predicted_strokes_to_finish_by_yardage = pd.DataFrame(np.hstack((yardage, predicted_strokes)))
    predicted_strokes_to_finish_by_yardage.columns = ['Distance to Pin (yards)', 'Pred. Strokes Left']

    return predicted_strokes_to_finish_by_yardage


def model_strokes_to_finish_from_par_5_tee_shot(par_5_tee_shot_features):
    """
    Trains a linear regresion predicting # shots to hole out after the tee shot, given DTP and
    the ball location.
    """
    strokes_model = LinearRegression()
    par_5_tee_shot_x = par_5_tee_shot_features.drop(['player', 'strokes_to_finish', 'strokes_from_here'], axis=1)
    par_5_tee_shot_y = par_5_tee_shot_features['strokes_to_finish']
    strokes_model.fit(par_5_tee_shot_x, par_5_tee_shot_y)

    coefficients = {feature: round(coef, 4) for feature, coef in zip(par_5_tee_shot_x.columns, strokes_model.coef_)}
    coefficients = pd.Series(coefficients).sort_values(ascending=True)

    return strokes_model, coefficients


def rank_players_by_strokes_to_finish_from_par_5_tee_shot(par_5_tee_shot_features, strokes_model):
    """
    Ranks players by their average predicted number of strokes to hole-out after the tee shot.
    """
    average_tee_shot_performance_by_player = \
        par_5_tee_shot_features.groupby('player').mean().drop(['strokes_from_here', 'strokes_to_finish'], axis=1)

    strokes_gained = \
        pd.Series({player: strokes for player, strokes in
                   zip(par_5_tee_shot_features.player.unique(),
                       strokes_model.predict(average_tee_shot_performance_by_player))})

    strokes_gained.sort_values(ascending=True, inplace=True)
    return strokes_gained


def plot_fir_against_driving_distance(pga_data_added, player_tournament_identifier=['player','tournament']):
    """
    Plot scatterplot of % Fairways hit in Reg vs Average Driving Distance, for each player in each tournament.
    """
    fairways = pga_data_added[pga_data_added.hole_par == 5].groupby(player_tournament_identifier).agg(
        {'fir': 'mean'}).reset_index()
    fairways = fairways.rename({'fir': 'average_fir'}, axis=1)
    fairways = pga_data_added.merge(fairways, how='inner', on=player_tournament_identifier)

    fairways_ = fairways[['average_fir', 'avg_driving_distance']].drop_duplicates()

    sns.lmplot('avg_driving_distance', 'average_fir', data=fairways_)
    set_size(8.5, 3)
    plt.xlabel('Average Driving Distance')
    plt.ylabel('% Fairways in Reg')
    plt.title('% Fairways in Regulation vs Average Driving Distance')
    plt.tight_layout()
    plt.show()


def present_par_5_tee_shot_ranking(predicted_strokes_to_finish_par_5_player):
    """
    Show Bryson's Par 5 Tee Shot Ranking by SG.
    """
    return pd.DataFrame(
        {'Predicted Strokes to Hole-Out after Tee Shot': predicted_strokes_to_finish_par_5_player.round(3),
         'Rank': 1 + np.arange(len(predicted_strokes_to_finish_par_5_player))}).iloc[8:11, :]

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS

shot_level = pd.read_csv(r'/Users/andre/Downloads/pga_tour_shot_level_data.csv', encoding='cp1252')
# shot_level = shot_level.iloc[:1000, :]
hole_identifier = ['player', 'round_number', 'hole_number', 'tournament']


def compute_total_strokes(shot_level):
    total_strokes = shot_level.groupby(hole_identifier)['hole_number'].transform(np.size)
    return total_strokes


def compute_gir(shot_level):
    def is_gir(hole_data):
        for index, shot in hole_data.iterrows():
            if (shot['shot_end_location'] == 'green') & (shot['shot_number'] <= shot['hole_par'] - 2):
                return 1
        return 0

    gir_by_hole = shot_level.groupby(hole_identifier).apply(
        lambda hole_data: is_gir(hole_data)).reset_index(name='gir')

    gir = shot_level.merge(gir_by_hole, on=hole_identifier, how='inner')['gir']

    return gir


shot_level['hole_yards'] = shot_level['hole_yards'].apply(lambda entry: float(entry.replace('yds', '')))
shot_level['shot_distance'] = shot_level['shot_start_distance_yards'] - shot_level['shot_end_distance_yards']


def simplify_shot_end_location(entry):
    remove_words = ['left ', 'right ', 'front ', 'center ']
    for remove_word in remove_words:
        entry = entry.replace(remove_word, '')
    entry = entry.replace(' ', '_')
    return entry


shot_level['total_strokes'] = compute_total_strokes(shot_level)
# shot_level['gir'] = compute_gir(shot_level)

shot_level['shot_end_location_simplified'] = shot_level['shot_end_location'].apply(
    lambda entry: simplify_shot_end_location(entry))

dataset = shot_level[shot_level.hole_par >= 4]

condition = [(shot_level['shot_number'] == 1) & (shot_level['hole_par'] == 5)][0]
features = ['hole_par', 'shot_end_distance_yards', 'shot_end_location_simplified']
x = shot_level[condition][features]
x.hole_par = x.hole_par.apply(str)
y = shot_level[condition]['total_strokes'] - 1

x = pd.get_dummies(data=x, drop_first=True)
x.head()
model = OLS(y, x)
results = model.fit()
print(results.params.sort_values(ascending=True))

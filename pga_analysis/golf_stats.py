"""
Compute golf-related metrics to be used in the analyses.
"""
import pandas as pd
import numpy as np

hole_identifier = ['player', 'round_number', 'hole_number', 'tournament']


def compute_total_strokes(pga_data: pd.DataFrame):
    """Returns the total strokes to complete the hole in the data."""
    total_strokes = pga_data.groupby(hole_identifier)['shot_number'].transform(max)
    return total_strokes


def compute_gir(pga_data: pd.DataFrame):
    """Returns 1 if a green in regulation was achieved on the hole, else 0."""

    def is_gir(hole_data):
        for index, shot in hole_data.iterrows():
            if (shot['shot_end_location'] == 'green') & (shot['shot_number'] <= shot['hole_par'] - 2):
                return 1
        return 0

    gir_by_hole = pga_data.groupby(hole_identifier).apply(
        lambda hole_data: is_gir(hole_data)).reset_index(name='gir')

    gir = pga_data.merge(gir_by_hole, on=hole_identifier, how='inner')['gir']

    return gir


def compute_fir(pga_data: pd.DataFrame):
    """
    For Par 4 & 5, returns 1 if fairway was hit in regulation on the hole, else 0.
    For Par 3, returns NaN.
    """

    def is_fir(hole_data):
        for index, shot in hole_data.iterrows():
            if shot['hole_par'] == 3:
                return np.NaN
            if ('fairway' in shot['shot_end_location']) & ('bunker' not in shot['shot_end_location']) & (
                    shot['shot_number'] == 1):
                return 1
        return 0

    fir_by_hole = pga_data.groupby(hole_identifier).apply(
        lambda hole_data: is_fir(hole_data)).reset_index(name='fir')

    fir = pga_data.merge(fir_by_hole, on=hole_identifier, how='inner')['fir']

    return fir


def hole_yards_to_float(pga_data: pd.DataFrame):
    """
    Returns the `hole_yards` column as a float.
    """
    hole_yards = pga_data['hole_yards'].apply(lambda entry: float(entry.replace('yds', '')))
    return hole_yards


def simplify_shot_end_location(shot_end_location):
    """
    Removes the prefixes: left, right, front, center from the shot_end_location, to simplify analysis later.
    """
    remove_words = ['left ', 'right ', 'front ', 'center ']
    for remove_word in remove_words:
        shot_end_location = shot_end_location.replace(remove_word, '')
    shot_end_location = shot_end_location.replace(' ', '_')
    return shot_end_location


def compute_total_putts(putting_data: pd.DataFrame):
    """
    Computes the total number of putts that were taken to complete the hole.
    A putt is assumed to be any shot that starts on the green.
    """
    total_putts = \
        putting_data.groupby(hole_identifier)['shot_number'].transform(lambda x: x.max() - x.min() + 1)
    return total_putts


def compute_strokes_from_here(pga_data: pd.DataFrame):
    """
    Computes the number of strokes *taken including the current shot* to complete the hole.
    """
    strokes_from_here = \
        pga_data.groupby(hole_identifier)['shot_number'].transform('max') - pga_data['shot_number'] + 1

    return strokes_from_here

def cast_winnings_to_float(pga_data_added):
    """
    Cast 'winnings' from a string to float in $.
    """
    return pga_data_added['winnings'].apply(lambda winning: float(winning.replace('$', '').replace(',', '')))


"""
Strokes gained analyses.
"""
from pga_analysis.putting import *
from pga_analysis.par_5_tee_shots import *
import warnings
from scipy.stats import wilcoxon
import logging


warnings.simplefilter(action='ignore', category=FutureWarning)


def prepare_tournament_data(tournament, pga_data_added):
    """
    Returns pga_data for a given tournament, for players that made the cut (played 4 rounds).
    """
    tournament_data = pga_data_added[pga_data_added.tournament == tournament]

    players_made_the_cut = tournament_data.groupby('player')['round_number'].max() == 4
    players_made_the_cut = pd.Series(players_made_the_cut[players_made_the_cut].index)
    tournament_data = tournament_data.merge(players_made_the_cut, how='inner', on='player')

    return tournament_data


def compute_total_putting_strokes_gained_by_player(tournament_data, putting_data):
    """
    Computes total putting strokes gained for each player in the data.
    """
    _, first_shot_on_green = prepare_putting_data(tournament_data)
    putting_strokes_gained = compute_putting_strokes_gained(putting_data, first_shot_on_green, by_player=True,
                                                            aggregation='sum')
    return putting_strokes_gained


def compute_total_par_5_strokes_gained_by_player(tournament_data, strokes_model, par_5_tee_shot_features, pga_data_added):
    """
    Computes total strokes gained from the tee shot on a Par 5 for each player in the data.
    """
    par_5_analysis_data = get_par_5_analysis_data(tournament_data, common_shot_end_locations(pga_data_added))
    par_5_model_features = par_5_analysis_data.drop(['player', 'strokes_from_here', 'strokes_to_finish'], axis=1)

    # Gets columns in the same order as when the model was trained
    orig_model_columns = \
        par_5_tee_shot_features.drop(['player', 'strokes_to_finish', 'strokes_from_here'], axis=1).columns
    for column in orig_model_columns:
        if column not in par_5_analysis_data.columns:
            par_5_model_features[column] = [0] * len(par_5_model_features)
    par_5_model_features = par_5_model_features[orig_model_columns]

    par_5_strokes_gained = strokes_model.predict(par_5_model_features)
    strokes_left = pd.DataFrame({'player': par_5_analysis_data.player,
                                 'strokes_to_finish': par_5_analysis_data.strokes_to_finish,
                                 'predicted_strokes_to_finish': par_5_strokes_gained})
    strokes_left['strokes_gained'] = strokes_left.predicted_strokes_to_finish - strokes_left.strokes_to_finish
    cum_strokes_gained = strokes_left.groupby('player')['strokes_gained'].sum()

    return cum_strokes_gained


def plot_strokes_gained_driving_vs_putting_by_winnings(putting_data, strokes_model,
                                                       par_5_tee_shot_features, pga_data_added,
                                                       winnings_rank_min=0, winnings_rank_max=5,
                                                       over_all_tournaments=False):
    """
    Strokes gained violin plot for driving (left half) vs putting (right half).

    winnings_rank_{min / max}: Min/max rank of tournament maximum winnings to plot.
    """
    most_winnings_tournaments = \
        pga_data_added.groupby('tournament').winnings.max().sort_values(ascending=False)[winnings_rank_min:winnings_rank_max].index

    if not over_all_tournaments:
        fig, ax = plt.subplots(figsize=(17, 6))

        putting_strokes_gained_list = []
        tee_shot_strokes_gained_list = []

        for tournament in most_winnings_tournaments:
            tournament_data = prepare_tournament_data(tournament, pga_data_added)
            putting_strokes_gained = compute_total_putting_strokes_gained_by_player(tournament_data, putting_data).iloc[:, 0]
            tee_shot_strokes_gained = compute_total_par_5_strokes_gained_by_player(tournament_data, strokes_model,
                                                                                 par_5_tee_shot_features, pga_data_added)

            putting_strokes_gained_list.append(putting_strokes_gained)
            tee_shot_strokes_gained_list.append(tee_shot_strokes_gained)


    else:
        fig, ax = plt.subplots(figsize=(7, 6))

        putting_strokes_gained_list = [compute_total_putting_strokes_gained_by_player(pga_data_added, putting_data).iloc[:, 0]]
        tee_shot_strokes_gained_list = [compute_total_par_5_strokes_gained_by_player(pga_data_added, strokes_model,
                                                                                   par_5_tee_shot_features, pga_data_added)]

    def violin_plot(data, half, width_factor=1.0):
        violin_plot = ax.violinplot(data, positions=np.arange(0, len(data)),
                                    showmeans=False, showextrema=False, showmedians=False, widths=0.5 * width_factor)
        for b in violin_plot['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            if half == 'left':
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color('b')
            elif half == 'right':
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color('r')
        return violin_plot

    violin_left = violin_plot(tee_shot_strokes_gained_list, half='left')
    width_factor = 0.7 if not over_all_tournaments else 0.85
    violin_right = violin_plot(putting_strokes_gained_list, half='right', width_factor=width_factor)

    ax.legend([violin_left['bodies'][0],violin_right['bodies'][0]],['DRIVING', 'PUTTING'])
    if not over_all_tournaments:
        ax.set_xticklabels(
            ['', 'THE PLAYERS CHAMPIONSHIP 2021\n$2.7M', 'PGA CHAMPIONSHIP 2020\n$2M', 'WGC CONCESSION 2021\n$1.8M'
                , 'WGC FEDEX ST JUDE INVITATIONAL 2020\n$1.8M', 'BMW CHAMPIONSHIP 2020\n$1.7M']
        )
    else:
        ax.set_xticks([])
    plt.ylabel('Strokes Gained')
    plt.title('Total Strokes Gained (Player)')


def plot_putting_strokes_gained_distribution(first_shot_to_green):
    """
    Plot distribution of average putting strokes gained per round, by player
    """
    putting_strokes_gained_by_player = first_shot_to_green.groupby(['player', 'tournament', 'round_number']) \
        ['strokes_gained'].sum().reset_index().groupby(['player']).strokes_gained.mean()
    grid = sns.kdeplot(putting_strokes_gained_by_player, shade=True)
    plt.xlim([-3, 3])
    plt.ylabel('')
    grid.set(yticks=[])
    set_size(10, 2)
    plt.title('Putting: Average Strokes Gained per Round, by Player')
    plt.xlabel('Strokes Gained')


def plot_strokes_gained_above_or_below_15_feet(putting_data, first_shot_to_green):
    """
    Scatter plot of each player's strokes gained for putts below or above 15 feet.
    """
    sg_below_15_feet_by_player = compute_putting_strokes_gained(putting_data, first_shot_to_green[
        first_shot_to_green.shot_start_distance_feet < 15], by_player=True)
    sg_above_15_feet_by_player = compute_putting_strokes_gained(putting_data, first_shot_to_green[
        first_shot_to_green.shot_start_distance_feet >= 15], by_player=True)
    sg_15_feet_putts_threshold = sg_below_15_feet_by_player.merge(sg_above_15_feet_by_player, on='player', how='inner')
    sg_15_feet_putts_threshold.columns = ['below_15_feet', 'above_15_feet']
    grid = sns.lmplot(x='below_15_feet', y='above_15_feet', data=sg_15_feet_putts_threshold)
    plt.xlabel('Avg SG BELOW 15-feet')
    plt.ylabel('Avg SG ABOVE 15-feet')
    plt.title('Small correlation between good SHORT vs LONG putters\n')
    plt.xlim([-0.2, 0.2])
    plt.ylim([-0.2, 0.2])
    set_size(5,4)
    grid.set(xticks=[-0.2, 0, 0.2], yticks=[-0.2, 0, 0.2])
    plt.tight_layout()


def drive_or_putt_for_dough(pga_data_added, putting_data, strokes_model, par_5_tee_shot_features):
    """
    Wilcoxon Signed Rank test to test whether players gained more strokes from putting vs driving.
    Unused due to insignificant result.
    """
    putting_sg = compute_total_putting_strokes_gained_by_player(pga_data_added, putting_data).iloc[:,0]
    driving_sg = compute_total_par_5_strokes_gained_by_player(pga_data_added, strokes_model, par_5_tee_shot_features, pga_data_added)

    statistic, p_value = wilcoxon(x=driving_sg, y=putting_sg, alternative='greater')

    logger = logging.getLogger()
    logger.info(f'For each player, PUTTING contributed more SG than PAR 5 TEE SHOTS.\n'
                f'p-value = {round(p_value,3)}')

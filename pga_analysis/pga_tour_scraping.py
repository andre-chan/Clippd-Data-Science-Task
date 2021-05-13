"""
This script scrapes winnings and average driving distance for each tournament from the PGA Tour Stats website:
https://www.pgatour.com/stats.html.

To run,
1. Download ChromeDriver for your OS and Chrome version: https://chromedriver.chromium.org/downloads.
2. Put the executable in this working directory `path/to/Clippd-Data-Science-Task/pga_analysis/    print(par_5_tee_shots)
chromedriver`

1. We use ChromeDriver from the selenium package to simulate an instance of Chrome that selects the tournament data.
2. We extract data from the HTML source using a Selector from the scrapy package.
"""

import pandas as pd
import numpy as np

from selenium import webdriver
from time import sleep
from scrapy import Selector

import os

path_to_chrome_driver = os.path.join(os.getcwd(), r'chromedriver')

def initialise_chrome_driver(path_to_chrome_driver):
    """
    Initialise Chrome Driver to control the browser.
    """
    chromeOptions = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path=path_to_chrome_driver, options=chromeOptions)
    driver.maximize_window()
    return driver


def get_tournament_winnings(selector, year, tournament):
    """
    Select tournament winnings from HTML of PGA Tour website using XPath.
    """
    players = []
    winnings = []
    years = []
    tournaments = []

    rows = selector.xpath("//tr[contains(@id, 'playerStatsRow')]")
    for row in rows:
        player = row.xpath(".//td[@class='player-name']/a/text()").get()
        players.append(player)
        winning = row.xpath(".//td[contains(text(), '$')]/text()").get()
        winnings.append(winning)
        years.append(year)
        tournaments.append(tournament)
    return players, winnings, years, tournaments


def get_driving_distances(resp, year, tournament):
    """
    Select average driving distance from HTML of PGA Tour website using XPath.
    """
    players = []
    driving_distances = []
    years = []
    tournaments = []

    rows = resp.xpath("//tr[contains(@id, 'playerStatsRow')]")
    for row in rows:
        player = row.xpath(".//td[@class='player-name']/a/text()").get()
        players.append(player)
        driving_distance = row.xpath(".//td[5]/text()").get()
        driving_distances.append(driving_distance)
        years.append(year)
        tournaments.append(tournament)

    return players, driving_distances, years, tournaments

def scrape_pga_winnings_and_drives(driver) -> pd.DataFrame:
    """
    Scrapes PGA winnings and average driving distances and saves to csv.

    1. Use ChromeDriver from the selenium package to simulate an instance of Chrome that selects the tournament data.
    2. Extract data from the HTML source using Selector from the scrapy package.
    """
    winnings_url = "https://www.pgatour.com/stats/stat.109.y2020.eon.t525.html"
    drives_url = "https://www.pgatour.com/content/pgatour/stats/stat.317.y2021.eon.t018.html"
    tournaments_mapping = pd.read_csv(r'../data/tournaments.csv')
    years_list = tournaments_mapping['Year'].tolist()
    tournaments_list = tournaments_mapping['Tournament'].tolist()

    players = []
    winnings = []
    years = []
    tournaments = []

    print('\nScraping Tournament Winnings\n')
    for event in range(tournaments_mapping.shape[0]):
        driver.get(winnings_url)
        sleep(2)
        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--season hasCustomSelect']").click()
        driver.find_element_by_xpath(
            "//div[@class='with-chevron']//option[text()='{}']".format(years_list[event])).click()
        sleep(3)
        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--period hasCustomSelect']").click()
        driver.find_element_by_xpath(
            "//div[@class='with-chevron']//option[text()='Tournament Only']").click()
        sleep(2)

        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--tournament hasCustomSelect']").click()
        driver.find_element_by_xpath("//div[@class='with-chevron']//option[text()='{}']".format(tournaments_list[event])).click()
        sleep(4)
        selector = Selector(text=driver.page_source)
        player, winning, year, tournament = get_tournament_winnings(selector, years_list[event], tournaments_list[event])
        players = players + player
        winnings = winnings + winning
        years = years + [year]
        tournaments = tournaments + tournament

    df_winnings = pd.DataFrame({
        'player': players,
        'Tournament': tournaments,
        'winnings': winnings,
        'year': years
    })

    players = []
    driving_distances = []
    years = []
    tournaments = []

    print("\nScraping Driving Distances\n")
    for event in range(tournaments_mapping.shape[0]):

        driver.get(drives_url)
        sleep(2)

        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--season hasCustomSelect']").click()
        driver.find_element_by_xpath(
            "//div[@class='with-chevron']//option[text()='{}']".format(years_list[event])).click()
        sleep(3)
        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--period hasCustomSelect']").click()
        driver.find_element_by_xpath(
            "//div[@class='with-chevron']//option[text()='Tournament Only']").click()
        sleep(2)
        driver.find_element_by_xpath(
            "//select[@class='statistics-details-select statistics-details-select--tournament hasCustomSelect']").click()
        driver.find_element_by_xpath("//div[@class='with-chevron']//option[text()='{}']".format(tournaments_list[event])).click()
        sleep(4)

        selector = Selector(text=driver.page_source)

        player, driving_distance, year, tournament = get_driving_distances(selector, years_list[event], tournaments_list[event])

        players = players + player
        driving_distances = driving_distances + driving_distance
        years = years + year
        tournaments = tournaments + tournament

    df_driving_distances = pd.DataFrame({
        'player': players,
        'Tournament': tournaments,
        'avg_driving_distance': driving_distances,
        'year': years
    })

    winnings_and_driving_distances_by_player = pd.merge(
        left=df_winnings, right=df_driving_distances, how='left',
        on=['player', 'Tournament', 'year'])

    winnings_and_driving_distances_by_player.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
    winnings_and_driving_distances_by_player = \
        winnings_and_driving_distances_by_player.merge(tournaments_mapping, how='left',
                                                       on=['year', 'Tournament'])
    winnings_and_driving_distances_by_player.to_csv(r'data/winnings_and_driving_distances_by_player.csv', index=False)

    return winnings_and_driving_distances_by_player

driver = initialise_chrome_driver(path_to_chrome_driver)
driver.implicitly_wait(10)
scrape_pga_winnings_and_drives(driver)
driver.close()
from typing import List, Generator, Iterator, Tuple
from pathlib import Path
from timeit import default_timer as timer
from time import sleep
from datetime import timedelta
import shutil
import json
import pickle
from tqdm.autonotebook import trange, tqdm
import pandas as pd

ALL_YEARS = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019)

# TODO: Add FILTERBLADE here
def get_all_logs(path: Path, years=None, progress_bar=True) -> List[Path]:

    if years is None:
        years = ALL_YEARS

    invalid_years = set(years) - set(ALL_YEARS)
    if invalid_years:
        raise Exception(f"INVALID YEARS: {invalid_years}")

    for year in path.iterdir():

        if int(year.stem) not in years:
            continue

        if progress_bar:
            listed = list(year.iterdir())
            all_logs = tqdm(listed, total=len(listed), desc=f"{year.stem}")
        else:
            all_logs = year.iterdir()

        for log in all_logs:
            yield log


def get_all_logs_annually(path: Path, years=None, progress_bar=True, filterblade_path: Path = None) -> Generator[Tuple[int, List[Path]], None, None]:
    """
    filterblade_path: 
        - Path to the log_game_data.parquet
        - Table with following headers: 
            - log_id
            - red
            - kui
            - ton-nan
            - sanma
            - soku
            - p0-dan
            - p1-dan
            - p2-dan
            - p3-dan
            - p0-rating
            - p1-rating
            - p2-rating
            - p3-rating
    """
    
    if years is None:
        years = ALL_YEARS

    invalid_years = set(years) - set(ALL_YEARS)
    if invalid_years:
        raise Exception(f"INVALID YEARS: {invalid_years}")

    valid_ids = None
    if filterblade_path:
        game_data = pd.read_parquet(filterblade_path, engine='fastparquet')  # Use `fastparquet` to preserve categorical data
        game_data = filter_logs(game_data)
        valid_ids = game_data.index
        
    filterblade = filterblade_path is not None

    for year in path.iterdir():

        if int(year.stem) not in years:
            continue

        listed = [log for log in year.iterdir() if log.stem in valid_ids] if filterblade else list(year.iterdir())
        if progress_bar:
            yield int(year.stem), tqdm(listed, total=len(listed), desc=f"{year.stem}")
        else:
            yield int(year.stem), listed


def remove_categories(path: Path, progress_bar=True):
    for year in path.iterdir():
        for categories in year.iterdir():
            if categories.is_dir():
                if progress_bar:
                    listed = list(categories.iterdir())
                    all_logs = tqdm(
                        listed, total=len(listed), desc=f"{year.stem}-{categories.stem}"
                    )
                else:
                    all_logs = categories.iterdir()

                for log in all_logs:
                    shutil.move(str(log), str(log.parent.parent))

                categories.rmdir()


def get_logs(path: Path, n_logs: int, years=None, progress_bar=True) -> List[Path]:

    if years is None:
        years = ALL_YEARS

    invalid_years = set(years) - set(ALL_YEARS)
    if invalid_years:
        raise Exception(f"INVALID YEARS: {invalid_years}")

    n = n_logs
    k = len(years)
    logs_left = [
        (n // k) + (1 if i < (n % k) else 0) for i in range(k)
    ]  # Distribute number of logs for each year

    for year in path.iterdir():

        if int(year.stem) not in years:
            continue

        current_logs_left = logs_left.pop(0)

        if progress_bar:
            listed = list(year.iterdir())
            length = (
                current_logs_left if current_logs_left < len(listed) else len(listed)
            )
            all_logs = tqdm(listed, total=length, desc=f"{year.stem}")
        else:
            all_logs = year.iterdir()

        for log in all_logs:
            if current_logs_left <= 0:
                break
            else:
                current_logs_left -= 1
                yield log


def test_reading_speed(path, years):
    """ Calculate the time needed to parse logs from .json to Python dict. """

    start_time = timer()

    for log in get_all_logs(path, years):
        log_json = json.load(log.open())

    sleep(0.01)
    print(
        "Time elapsed (hh:mm:ss.ms) {:0>8}".format(
            str(timedelta(seconds=timer() - start_time))
        )
    )
    

def filter_logs(dataframe: pd.DataFrame, sanma=False, red=True, kui=True, soku=False, ton_nan=True, min_dan=16, min_rating=2000.0) -> pd.DataFrame:
    """
    ## Config Legend:
        - 'sanma' == 3-Player
        - 'red' == Contains red fives
        - 'kui' == Open-tanyao ([Kuitan](http://arcturus.su/wiki/Tanyao#Kuitan))
        - 'soku' == Fast rounds
        - 'ton-nan' == East-South
    """
    
    mask_sanma = dataframe['sanma'] if sanma else ~dataframe['sanma']
    mask_red = dataframe['red'] if red else ~dataframe['red']
    mask_kui = dataframe['kui'] if kui else ~dataframe['kui']
    mask_soku = dataframe['soku'] if soku else ~dataframe['soku']
    mask_ton_nan = dataframe['ton-nan'] if ton_nan else ~dataframe['ton-nan']
    mask_dan = (
        (dataframe['p0-dan'] >= min_dan) & 
        (dataframe['p1-dan'] >= min_dan) &
        (dataframe['p2-dan'] >= min_dan) &
        (dataframe['p3-dan'] >= min_dan)
    )
    
    mask_rating = (
        (dataframe['p0-rating'] >= min_rating) & 
        (dataframe['p1-rating'] >= min_rating) &
        (dataframe['p2-rating'] >= min_rating) &
        (dataframe['p3-rating'] >= min_rating)
    )

    return dataframe[
        mask_sanma &
        mask_red &
        mask_kui &
        mask_soku &
        mask_ton_nan &
        mask_dan &
        mask_rating
    ]

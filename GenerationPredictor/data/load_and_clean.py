from common_functions.shared_imports import *
from common_functions.log import logger

def load_and_clean_data(self,
                        csv_path: str,
                        start_date: str = '2015-01-01',
                        min_data_points: int = 8760):
    """Load and clean data, skip duplicate rows, and preserve most data while handling gaps."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    logger.info("Loading generation data...")
    # read CSV and parse dates
    df = pd.read_csv(path, parse_dates=['Datetime'])
    self.fuel_types = [c for c in df.columns if c != 'Datetime']

    # drop every duplicate timestamp (keep first of each pair)
    df = df.loc[~df['Datetime'].duplicated(keep='first')]

    # set index, sort, and filter by start date
    df = (df
        .set_index('Datetime')
        .sort_index()
        .loc[start_date:])

    if len(df) < min_data_points:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_data_points}")

    # clip extreme outliers (0.1st and 99.9th percentiles)
    for fuel in self.fuel_types:
        lo = df[fuel].quantile(0.001)
        hi = df[fuel].quantile(0.999)
        df[fuel] = df[fuel].clip(lower=max(0, lo), upper=hi)

    # enforce hourly frequency
    df = df.asfreq('h')

    # looser interpolation: fill gaps up to 24 hours
    df.interpolate(method='time', limit=24, inplace=True)
    df.ffill(limit=24, inplace=True)
    df.bfill(limit=24, inplace=True)

    # drop rows with more than 30% of fuel columns missing
    thresh = int(len(self.fuel_types) * 0.7)
    before_drop = len(df)
    df.dropna(thresh=thresh, inplace=True)
    logger.info(f"Removed {before_drop - len(df)} rows with >30% missing")

    # fill any small remaining gaps (up to 2 hours)
    df.ffill(limit=2, inplace=True)
    df.bfill(limit=2, inplace=True)

    # warn if any NaNs remain
    total_nans = df.isna().sum().sum()
    if total_nans > 0:
        logger.warning(f"{total_nans} NaNs remain after filling — check your data gaps")

    # compute total generation
    df['Total'] = df[self.fuel_types].sum(axis=1)

    # add small noise to each fuel series to avoid perfect correlations
    np.random.seed(self.random_state)
    for fuel in self.fuel_types:
        noise_std = df[fuel].std() * 0.001
        df[fuel] = (df[fuel] + np.random.normal(0, noise_std, len(df))).clip(lower=0)

    # recalculate total after noise
    df['Total'] = df[self.fuel_types].sum(axis=1)

    # final check on minimum length
    if len(df) < min_data_points:
        raise ValueError(f"After cleaning: {len(df)} rows, need at least {min_data_points}")

    self.generation_data = df
    logger.info(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")
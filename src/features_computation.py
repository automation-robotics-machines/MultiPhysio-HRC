import logging
from tqdm import tqdm
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Tuple

import polars as pl
from physio_features import *
from utils.utils import *
from utils.logger import *
import warnings
warnings.filterwarnings("ignore")
bad = load_config('config/config.json')["BAD WINDOWS"]
modified_data = {tuple(sublist[0]): sublist[1] for sublist in bad}
BAD_WINDOWS = pd.DataFrame(index=pd.MultiIndex.from_tuples(modified_data.keys()), columns=["Values"])
BAD_WINDOWS["Values"] = [sublist[1] for sublist in bad]
BAD_WINDOWS_pl = pl.from_pandas(BAD_WINDOWS.reset_index())
FILE_PATH = '/media/andreabussolan/One Touch/data/physio/mem'
SAVE_PATH = './data/features/mem'


class BioFeaturesExtractor:
    def __init__(
            self,
            data: Union[pd.DataFrame, pl.DataFrame],
            subjects: List[str],
            scenarios: List[str],
            markers: bool = False,
            window: int = 60,
            fs: Union[int, None] = None,
            use_polars: bool = False
    ) -> None:

        self.use_polars = use_polars
        self.data = data
        self.subjects = subjects
        self.scenarios = scenarios
        self.window = window
        self.markers = markers
        self.fs = fs if fs is not None else 256
        self.logger = CustomLogger().get_logger(__name__)

        self.SIGNALS = ["ECG", "EDA", "EMG", "RESP"]  # Correct order!
        # self.SIGNALS = ["ECG", "EDA", "RESP"]  # Correct order!
        if self.markers:
            self.SIGNALS = self.SIGNALS + ['UserMarker']
        # self.data = self.data[['ID', 'Class', 'Repetition'] + self.SIGNALS + ['multi_index']]
        self.semaphore = threading.Semaphore()
        self.worker_threads = {}
        self.holder = {"ECG": [], "EDA": [], "EMG": [], "RESP": []}
        self.locks = {data_type: threading.Lock() for data_type in self.SIGNALS}
        suffix = '_markers' if self.markers else ''
        self.save_name = f'bio_features_{self.window}s{suffix}.csv'

        self.feature_function = {
            "ECG": (partial(extract_hrv_nk_features, fs=self.fs), 1),
            "EDA": (partial(extract_eda_time_and_frequency_features, fs=self.fs, window=self.window), 2),
            "EMG": (partial(extract_emg_features, fs=self.fs), 3),
            "RESP": (partial(extract_resp_features, fs=self.fs), 4),
        }
        # self.feature_function = {
        #     "ECG": (partial(extract_hrv_nk_features, fs=self.fs), 0),
        #     "EDA": (partial(extract_eda_time_and_frequency_features, fs=self.fs, window=self.window), 1),
        #     "RESP": (partial(extract_resp_features, fs=self.fs), 2),
        # }

    def run(self) -> Union[pd.DataFrame, pl.DataFrame]:
        max_workers = min(5, len(self.subjects))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.compute_subject, subject)
                for subject in self.subjects
            ]

        for future in as_completed(futures):
            future.result()

        features = (
            pd.concat(
                [pd.concat(self.holder[data_type], axis=0) for data_type in self.SIGNALS if data_type != 'UserMarker'],
                 axis=1)
            .reset_index()
        )

        start = time.time_ns()
        if not self.use_polars:
            features.to_csv(os.path.join(SAVE_PATH, self.save_name))
        else:
            features = pl.from_pandas(features)
            features.write_csv(os.path.join(SAVE_PATH, self.save_name))
        print('Save bio: ', (time.time_ns() - start) * 1e-6)
        return features

    def compute_subject(self, subject: str) -> None:
        start = time.time_ns()
        for scenario in self.scenarios:
            self.logger.info(f"Processing {subject} in {scenario}")
            try:
                if not self.use_polars:
                    if (subject, scenario) not in self.data.index:
                        self.logger.warning(f"No data for subject {subject} in scenario {scenario}")
                        continue
                    reps = (
                        self.data.loc[(subject, scenario, slice(None)), :]
                        .index.get_level_values(2).unique().to_list())
                else:
                    if self.data.filter((pl.col("ID") == subject) & (pl.col("Class") == scenario)).is_empty():
                        self.logger.warning(f"No data for subject {subject} in scenario {scenario}")
                        continue
                    reps = (self.data.filter((pl.col("ID") == subject) & (pl.col("Class") == scenario))["Repetition"]
                            .unique()
                            .to_list())

                for rep in reps:
                    self.process_repetition(subject, scenario, rep)
            except Exception as e:
                self.logger.error(f"Error processing subject {subject} in scenario {scenario}: {str(e)}")

        self.logger.info(f'Computed features for subject {subject} in [ms]: {(time.time_ns() - start) * 1e-6}')

    def process_repetition(self, subject: str, scenario: str, rep: Union[int, str]) -> None:
        for data_type in self.SIGNALS:
            try:
                fun, cols = self.feature_function[data_type]
                if not self.use_polars:
                    data_scenario = self.data.loc[(subject, scenario, rep), :].iloc[:, cols].to_numpy()
                else:
                    data_scenario = (
                        self.data.filter(
                            (pl.col("ID") == subject) &
                            (pl.col("Class") == scenario) &
                            (pl.col("Repetition") == rep)
                        )
                        .select(pl.all().exclude(["ID", "Class", "Repetition"]))
                        .to_numpy()[:, cols]
                        .astype(float)
                    )

                if self.markers:
                    marker_indexes = self.get_marked_indexes(subject, scenario, rep)
                    if len(marker_indexes) < 2:
                        print(f"Not enough markers for subject {subject}, scenario {scenario}, repetition {rep}")
                        continue

                    data_windowed = [
                        data_scenario[marker_indexes[j]:marker_indexes[j + 1]]
                        for j in range(0, len(marker_indexes) - 1, 2)
                    ]
                else:
                    data_windowed = create_windows(data_scenario, self.window, self.fs)

                # if not self.markers:
                #     self.sanitize_bad_windows(data_windowed, subject, scenario, rep)

                for j, w in enumerate(data_windowed):
                    features = fun(w)
                    features['ID'] = [subject]
                    features['Class'] = [scenario]
                    features['Repetition'] = [rep]
                    features['Window'] = [j]
                    features = features.set_index(['ID', 'Class', 'Repetition', 'Window'])

                    with self.semaphore:
                        self.holder[data_type].append(features)

            except Exception as e:
                self.logger.error(f"Error processing {data_type} for subject {subject},"
                                  f" scenario {scenario}, rep {rep}: {str(e)}")

    def sanitize_bad_windows(self, data_windowed, subject: str, scenario: str, rep: Union[int, str]):
        if not self.use_polars:
            if (subject, scenario, rep) in BAD_WINDOWS.index:
                skip = BAD_WINDOWS.loc[(subject, scenario, rep), "Values"]
                data_windowed = np.delete(data_windowed, skip, axis=0)
        else:
            bad_windows = BAD_WINDOWS_pl.filter(
                (pl.col("level_0") == subject) &
                (pl.col("level_1") == scenario) &
                (pl.col("level_2") == rep)
            )
            if not bad_windows.is_empty():
                skip = bad_windows.select("Values").to_numpy().flatten()
                data_windowed = np.delete(data_windowed, skip, axis=0)

        return data_windowed

    def get_marked_indexes(self, subject: str, scenario: str, rep: Union[int, str]) -> List[Tuple[int, int]]:
        if not self.use_polars:
            marker_intervals = self.data.loc[(subject, scenario, rep), 'UserMarker'].dropna().tolist()
        else:
            marker_intervals = (
                (
                    self.data
                    .filter(
                        (pl.col("ID") == subject) &
                        (pl.col("Class") == scenario) &
                        (pl.col("Repetition") == rep)
                    )
                    .with_row_index()
                )
                .filter((pl.col("UserMarker") == 1))
                .select(pl.col("index"))
                .to_series()
                .to_list()
            )
        return marker_intervals


class EEGFeaturesExtractor:
    def __init__(
            self,
            data: Union[pd.DataFrame, pl.DataFrame],
            subjects: List[str],
            scenarios: List[str],
            markers: bool = False,
            fs: Union[int, None] = None,
            window: int = 5,
            use_polars: bool = True
    ) -> None:

        self.use_polars = use_polars
        self.data = data
        self.scenarios = scenarios
        self.subjects = subjects
        self.markers = markers
        self.window = window

        self.plv_matrix_ls = []
        self.corr_matrix_ls = []
        self.fs = fs if fs is not None else 256
        suffix = '_markers' if self.markers else ''
        self.save_name1 = f'eeg_features_for_plots_{self.window}s{suffix}.csv'
        self.save_name2 = f'eeg_features_{self.window}s{suffix}.csv'

    @timing
    def run(self) -> tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame]]:
        all_features = []
        all_features_channels_as_columns = []
        result = None
        max_workers = min(5, len(self.subjects))
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [
                executor.submit(self.compute_features, subject)
                for subject in self.subjects
            ]

        for future in futures:
            result = future.result()
            self.corr_matrix_ls.extend(result[0])
            self.plv_matrix_ls.extend(result[1])
            all_features.extend(result[2])
            all_features_channels_as_columns.extend(result[3])

        features_for_plots = pd.concat(all_features, axis=0)
        features_for_plots.to_csv(os.path.join(SAVE_PATH, self.save_name1), index=False)

        cols = ['ID', 'Class', 'Repetition', 'Window'] + \
            [ch + '_' + ft
             for ft in result[2][0].columns[1:-4]
             for ch in result[2][0]['Channel'].unique().tolist()] + \
            ['ThF3/AlP3', 'ThF3/AlP4', 'ThF4/AlP4', 'ThF4/AlP3']
        features_df = pd.DataFrame(all_features_channels_as_columns, columns=cols)

        if self.use_polars:
            features_df = pl.from_pandas(features_df)
            features_df.write_csv(os.path.join(SAVE_PATH, self.save_name2))
        else:
            features_df.to_csv(os.path.join(SAVE_PATH, self.save_name2), index=False)

        with open('./data/corr.pkl', 'wb') as f:
            pickle.dump(self.corr_matrix_ls, f)

        with open('./data/plv.pkl', 'wb') as f:
            pickle.dump(self.plv_matrix_ls, f)

        return features_df, features_for_plots

    def compute_features(
            self,
            subject: str,
    ) -> tuple[List, List, List, List]:

        corr_matrix_ls = []
        plv_matrix_ls = []
        all_features = []
        all_features_channels_as_columns = []
        start = time.time_ns()
        for scenario in self.scenarios:
            if self.use_polars:
                data_scenario = (
                    self.data.filter((pl.col("ID") == subject) & (pl.col("Class") == scenario))
                    .select('Repetition', pl.selectors.starts_with('EEG'))
                )
                if data_scenario.is_empty():
                    print(f"No data for subject {subject} in scenario {scenario}")
                    continue
                reps = data_scenario["Repetition"].unique().to_list()
            else:
                if (subject, scenario) not in self.data.index:
                    print(f"No data for subject {subject} in scenario {scenario}")
                    continue
                data_scenario = self.data.loc[(subject, scenario, slice(None)), :].iloc[:, -12:]
                reps = data_scenario.index.get_level_values(2).unique().to_list()

            for rep in reps:
                if self.use_polars:
                    data_rep = (data_scenario
                                .filter(pl.col("Repetition") == rep)
                                .select(pl.selectors.starts_with('EEG'))
                                .to_numpy())
                else:
                    data_rep = data_scenario.loc[(subject, scenario, rep)].to_numpy()

                if self.markers:
                    marker_indexes = (
                        (
                            self.data
                            .filter(
                                (pl.col("ID") == subject) &
                                (pl.col("Class") == scenario) &
                                (pl.col("Repetition") == rep)
                            )
                            .with_row_index()
                        )
                        .filter((pl.col("UserMarker") == 1))
                        .select(pl.col("index"))
                        .to_series()
                        .to_list()
                    )

                    if len(marker_indexes) < 2:
                        print(f"Not enough markers for subject {subject}, scenario {scenario}, repetition {rep}")
                        continue

                    segments = [
                        data_rep[marker_indexes[j]:marker_indexes[j + 1], :]
                        for j in range(0, len(marker_indexes) - 1, 2)
                    ]
                else:
                    segments = create_windows(data_rep, self.window, self.fs)

                for j, segment in enumerate(tqdm(segments)):
                    self.process_segment(segment, subject, scenario, rep, j,
                                         corr_matrix_ls, plv_matrix_ls,
                                         all_features, all_features_channels_as_columns)

        print(f'Computed EEG feats for {subject} in [ms]: ', (time.time_ns() - start) * 1e-6)
        return corr_matrix_ls, plv_matrix_ls, all_features, all_features_channels_as_columns

    def process_segment(
            self,
            segment,
            subject,
            scenario,
            rep,
            j,
            corr_matrix_ls,
            plv_matrix_ls,
            all_features,
            all_features_channels_as_columns
    ):
        """Processes a single EEG segment: feature extraction and storage."""
        features, ratios, info = extract_eeg_features(segment, self.fs)

        corr_matrix_ls.append([subject, j, scenario, rep, pearson_correlation(segment)])
        plv_matrix_ls.append([subject, j, scenario, rep, phase_locking_value(segment)])

        features["Repetition"] = [rep] * info["shape"][0]
        features["Class"] = [scenario] * info["shape"][0]
        features["Window"] = [j] * info["shape"][0]
        features["ID"] = [subject] * info["shape"][0]

        features_channels_columns = ([subject, scenario, rep, j] +
                                     features.iloc[:, 1:-4].values.flatten().tolist() +
                                     ratios)

        all_features_channels_as_columns.append(features_channels_columns)
        all_features.append(features)


class FeaturesExtractor:
    def __init__(
            self,
            data: pd.DataFrame | None,
            path: str,
            fs: int | None = None,
            use_polars: bool = False,
            markers: bool = False,
    ) -> None:

        self.path = path
        self.fe_bio, self.fe_eeg = None, None
        self.scenarios, self.subjects = None, None
        self.corr_matrix_ls, self.plv_matrix_ls = [], []
        self.fs = fs if fs is not None else 256
        self.markers = markers

        if data is not None:
            if not use_polars:
                self.data = data.to_pandas()
                self.data = (
                    self.data
                    .set_index(["ID", "Class", "Repetition"])
                    .sort_index()
                    .drop(columns=[""])
                )
                self.subjects = [str(sub) for sub in self.data.index.get_level_values(0).unique().to_list()]
                self.scenarios = self.data.index.get_level_values(1).unique().to_list()
            else:
                self.data = data.with_columns(
                    pl.concat_str(["ID", "Class", "Repetition"], separator="_").alias("multi_index")
                ).set_sorted("multi_index")
                self.data = self.data.drop("multi_index")
                self.subjects = [str(sub) for sub in self.data["ID"].unique().to_list()]
                self.scenarios = self.data["Class"].unique().to_list()
        else:
            try:
                self.data = pd.read_csv(self.path, index_col=0) \
                    .set_index(["ID", "Class", "Repetition"]) \
                    .sort_index()

                self.subjects = [str(sub) for sub in self.data.index.get_level_values(0).unique().to_list()]
                self.scenarios = self.data.index.get_level_values(1).unique().to_list()

            except FileNotFoundError:
                print('File not found...')

        self.fe_bio = BioFeaturesExtractor(
            self.data, subjects=self.subjects, scenarios=self.scenarios,
            fs=self.fs, use_polars=True, markers=self.markers
        )
        self.fe_eeg = EEGFeaturesExtractor(
            self.data, subjects=self.subjects, scenarios=self.scenarios,
            fs=self.fs, use_polars=True, markers=self.markers
        )

    @timing
    def run(self, only_bio: bool = False, only_eeg: bool = False) -> (
            pd.DataFrame |
            tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]
    ):
        with ThreadPoolExecutor() as executor:
            if only_bio:
                feats_bio = executor.submit(self.fe_bio.run)
                return feats_bio
            elif only_eeg:
                feats_eeg = executor.submit(self.fe_eeg.run)
                return feats_eeg
            else:
                feats_bio = executor.submit(self.fe_bio.run)
                feats_eeg = executor.submit(self.fe_eeg.run)
                return feats_bio.result(), feats_eeg.result()


def merge_data(save: bool = True) -> pd.DataFrame:
    files = [file for file in import_filenames(FILE_PATH)[0] if 'filtered' in file]
    data = pl.concat(
        [pl.read_csv(os.path.join(FILE_PATH, subject_fold)) for subject_fold in files if 'all_' not in subject_fold],
        how='vertical'
    ).with_columns(
        [pl.col("ID").cast(pl.Utf8),
         pl.col("Class").cast(pl.Utf8)]
    )
    if save:
        data.write_csv(os.path.join(FILE_PATH, "all_filtered.csv"))
    return data


if __name__ == "__main__":
    MERGE = True
    if MERGE:
        merged = merge_data(save=False)
    else:
        merged = None

    path = os.path.join(FILE_PATH, "all_filtered.csv")
    fe = FeaturesExtractor(data=merged, fs=256, path=path, use_polars=True, markers=False)
    fe.run(only_bio=True)

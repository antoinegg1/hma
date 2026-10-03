"""Frozen task catalogs for the native MLE profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Suite = Literal[
    "mlebench_lite",
    "mlebench_medium",
    "mlebench_high",
    "mledojo_unique",
]
DataSource = Literal["mlebench", "dojo_dsbench", "dojo_kaggle"]
FeedbackMode = Literal["blind", "interactive"]

MLEBENCH_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
MLEDOJO_COMMIT = "667a8fd0e959eb47f140ce7e621f831fdb55e5dd"
SNAPSHOT_ACTIVE_SECONDS = (43_200, 86_400)


@dataclass(frozen=True)
class MLETaskSpec:
    """One frozen task and its native evaluation contract."""

    experiment_id: str
    task_name: str
    suite: Suite
    slug: str
    title: str
    metric: str
    higher_is_better: bool
    category: str
    data_source: DataSource
    feedback_mode: FeedbackMode
    submission_limit: int | None

    @property
    def upstream_commit(self) -> str:
        """Returns the source revision that owns this task's grader."""
        return MLEBENCH_COMMIT if self.is_mlebench else MLEDOJO_COMMIT

    @property
    def is_mlebench(self) -> bool:
        """Returns whether this task uses the pinned MLE-bench adapter."""
        return self.data_source == "mlebench"

    @property
    def source_name(self) -> str:
        """Returns the data-root source checkout name."""
        return "mle-bench" if self.is_mlebench else "mle-dojo"

    @property
    def data_suffix(self) -> str:
        """Returns the task-local prepared-data directory name."""
        return "prepared" if self.is_mlebench else "data"

    @property
    def agent_metric(self) -> str:
        """Returns a metric label that cannot reveal a competition slug."""
        return self.metric.replace(self.slug, "task")

    @property
    def data_subpath(self) -> str:
        """Returns the neutral data-root-relative task directory."""
        return f"datasets/{self.task_name}"

    @property
    def legacy_data_subpath(self) -> str:
        """Returns the pre-adapter-migration directory used only by preparation."""
        return f"{self.source_name}/{self.slug}"

    @property
    def public_subpath(self) -> str:
        """Returns the data-root-relative public directory."""
        return f"{self.data_subpath}/{self.data_suffix}/public"


_MBL_ROWS = (
    (
        "aerial-cactus-identification",
        "Aerial Cactus Identification",
        "auc-roc",
        True,
        "CV",
    ),
    (
        "aptos2019-blindness-detection",
        "APTOS 2019 Blindness Detection",
        "quadratic-weighted-kappa",
        True,
        "CV",
    ),
    (
        "denoising-dirty-documents",
        "Denoising Dirty Documents",
        "root-mean-squared-error",
        False,
        "CV",
    ),
    (
        "detecting-insults-in-social-commentary",
        "Detecting Insults in Social Commentary",
        "auc-roc",
        True,
        "NLP",
    ),
    (
        "dog-breed-identification",
        "Dog Breed Identification",
        "multi-class-log-loss",
        False,
        "CV",
    ),
    (
        "dogs-vs-cats-redux-kernels-edition",
        "Dogs vs. Cats Redux: Kernels Edition",
        "log-loss",
        False,
        "CV",
    ),
    (
        "histopathologic-cancer-detection",
        "Histopathologic Cancer Detection",
        "auc-roc",
        True,
        "CV",
    ),
    (
        "jigsaw-toxic-comment-classification-challenge",
        "Toxic Comment Classification Challenge",
        "column-wise-roc-auc",
        True,
        "NLP",
    ),
    (
        "leaf-classification",
        "Leaf Classification",
        "multi-class-log-loss",
        False,
        "Tabular",
    ),
    (
        "mlsp-2013-birds",
        "MLSP 2013 Bird Classification Challenge",
        "auc-roc",
        True,
        "Audio",
    ),
    (
        "new-york-city-taxi-fare-prediction",
        "New York City Taxi Fare Prediction",
        "root-mean-squared-error",
        False,
        "Tabular",
    ),
    (
        "nomad2018-predict-transparent-conductors",
        "Nomad2018 Predicting Transparent Conductors",
        "mean-column-wise-rmsle",
        False,
        "Tabular",
    ),
    (
        "plant-pathology-2020-fgvc7",
        "Plant Pathology 2020 - FGVC7",
        "mean-column-wise-roc-auc",
        True,
        "CV",
    ),
    ("random-acts-of-pizza", "Random Acts of Pizza", "auc-roc", True, "NLP"),
    (
        "ranzcr-clip-catheter-line-classification",
        "RANZCR CLiP - Catheter and Line Position Challenge",
        "auc-roc",
        True,
        "CV",
    ),
    (
        "siim-isic-melanoma-classification",
        "SIIM-ISIC Melanoma Classification",
        "auc-roc",
        True,
        "CV",
    ),
    (
        "spooky-author-identification",
        "Spooky Author Identification",
        "multi-class-log-loss",
        False,
        "NLP",
    ),
    (
        "tabular-playground-series-dec-2021",
        "Tabular Playground Series - Dec 2021",
        "classification-accuracy",
        True,
        "Tabular",
    ),
    (
        "tabular-playground-series-may-2022",
        "Tabular Playground Series - May 2022",
        "auc-roc",
        True,
        "Tabular",
    ),
    (
        "text-normalization-challenge-english-language",
        "Text Normalization Challenge - English",
        "accuracy",
        True,
        "NLP",
    ),
    (
        "text-normalization-challenge-russian-language",
        "Text Normalization Challenge - Russian",
        "accuracy",
        True,
        "NLP",
    ),
    (
        "the-icml-2013-whale-challenge-right-whale-redux",
        "ICML 2013 Right Whale Redux",
        "auc-roc",
        True,
        "Audio",
    ),
)

MLEBENCH_LITE = tuple(
    MLETaskSpec(
        experiment_id=f"MBL-{index:02d}",
        task_name=f"mbl_{index:02d}",
        suite="mlebench_lite",
        slug=slug,
        title=title,
        metric=metric,
        higher_is_better=higher_is_better,
        category=category,
        data_source="mlebench",
        feedback_mode="blind",
        submission_limit=None,
    )
    for index, (slug, title, metric, higher_is_better, category) in enumerate(_MBL_ROWS, start=1)
)

# Cold-start reservations are sized from each prepared public dataset. They are
# admission shares, not task ceilings: every MLE-bench Lite agent can still
# borrow up to its fixed 12-CPU/96-GiB hard limit.
_MBL_SMALL = {"01", "03", "04", "08", "09", "12", "14", "17"}
_MBL_LARGE = {"02", "07", "11", "15"}
MLEBENCH_LITE_RESERVATIONS = {
    f"mbl_{index:02d}": (
        (2, 16_384)
        if f"{index:02d}" in _MBL_SMALL
        else (6, 49_152)
        if f"{index:02d}" in _MBL_LARGE
        else (8, 65_536)
        if index == 16
        else (4, 24_576)
    )
    for index in range(1, 23)
}

_MBM_ROWS = (
    (
        "AI4Code",
        "Google AI4Code - Understand Code in Python Notebooks",
        "kendall-tau",
        True,
        "Text Classification",
    ),
    (
        "alaska2-image-steganalysis",
        "ALASKA2 Image Steganalysis",
        "weighted-auroc",
        True,
        "Image Classification",
    ),
    (
        "billion-word-imputation",
        "Billion Word Imputation",
        "levenshtein-distance",
        False,
        "Training LLMs",
    ),
    (
        "cassava-leaf-disease-classification",
        "Cassava Leaf Disease Classification",
        "accuracy",
        True,
        "Image Classification",
    ),
    (
        "cdiscount-image-classification-challenge",
        "Cdiscount’s Image Classification Challenge",
        "accuracy",
        True,
        "Image Classification",
    ),
    (
        "chaii-hindi-and-tamil-question-answering",
        "Hindi and Tamil Question Answering",
        "word-level-jaccard-score",
        True,
        "Training LLMs",
    ),
    (
        "champs-scalar-coupling",
        "Predicting Molecular Properties",
        "log-mean-absolute-error",
        False,
        "Tabular",
    ),
    (
        "facebook-recruiting-iii-keyword-extraction",
        "Facebook Recruiting III - Keyword Extraction",
        "micro-f1-score",
        True,
        "Text Classification",
    ),
    (
        "freesound-audio-tagging-2019",
        "Freesound Audio Tagging 2019",
        "label-ranking-average-precision",
        True,
        "Audio Classification",
    ),
    (
        "google-quest-challenge",
        "Google QUEST Q&A Labeling",
        "column-wise-spearman",
        True,
        "Training LLMs",
    ),
    (
        "h-and-m-personalized-fashion-recommendations",
        "H&M Personalized Fashion Recommendations",
        "MAP@12",
        True,
        "Tabular",
    ),
    (
        "herbarium-2020-fgvc7",
        "Herbarium 2020 - FGVC7",
        "macro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "herbarium-2021-fgvc8",
        "Herbarium 2021 - FGVC8",
        "macro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "herbarium-2022-fgvc9",
        "Herbarium 2022 - FGVC9",
        "macro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "hotel-id-2021-fgvc8",
        "Hotel-ID to Combat Human Trafficking 2021 - FGVC8",
        "map-at-5",
        True,
        "Image Classification",
    ),
    (
        "hubmap-kidney-segmentation",
        "HuBMAP - Hacking the Kidney",
        "dice-coefficient",
        True,
        "Image Segmentation",
    ),
    (
        "icecube-neutrinos-in-deep-ice",
        "IceCube - Neutrinos in Deep Ice",
        "mean-angular-error",
        False,
        "Tabular",
    ),
    (
        "imet-2020-fgvc7",
        "iMet Collection 2020 - FGVC7",
        "micro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "inaturalist-2019-fgvc6",
        "iNaturalist 2019 at FGVC6",
        "top-1-classification-error",
        False,
        "Image Classification",
    ),
    (
        "iwildcam-2020-fgvc7",
        "iWildCam 2020 - FGVC7",
        "accuracy",
        True,
        "Image Classification",
    ),
    (
        "jigsaw-unintended-bias-in-toxicity-classification",
        "Jigsaw Unintended Bias in Toxicity Classification",
        "jigsaw-unintended-bias-in-toxicity-classification-score",
        True,
        "Text Classification",
    ),
    (
        "kuzushiji-recognition",
        "Kuzushiji Recognition",
        "f1-score",
        True,
        "Image Classification",
    ),
    (
        "learning-agency-lab-automated-essay-scoring-2",
        "Learning Agency Lab - Automated Essay Scoring 2.0",
        "quadratic-weighted-kappa",
        True,
        "Text Classification",
    ),
    (
        "lmsys-chatbot-arena",
        "LMSYS - Chatbot Arena Human Preference Predictions",
        "multi-class-log-loss",
        False,
        "Text Classification",
    ),
    (
        "multi-modal-gesture-recognition",
        "Multi-modal Gesture Recognition",
        "levenhstein-distance",
        False,
        "Image Segmentation",
    ),
    (
        "osic-pulmonary-fibrosis-progression",
        "OSIC Pulmonary Fibrosis Progression",
        "modified-laplace-log-likelihood",
        True,
        "Forecasting",
    ),
    (
        "petfinder-pawpularity-score",
        "PetFinder.my - Pawpularity Contest",
        "root-mean-squared-error",
        False,
        "Image (Other)",
    ),
    (
        "plant-pathology-2021-fgvc8",
        "Plant Pathology 2021 - FGVC8",
        "micro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "seti-breakthrough-listen",
        "SETI Breakthrough Listen - E.T. Signal Search",
        "auc-roc",
        True,
        "Signal Processing",
    ),
    (
        "statoil-iceberg-classifier-challenge",
        "Statoil/C-CORE Iceberg Classifier Challenge",
        "log-loss",
        False,
        "Image Classification",
    ),
    (
        "tensorflow-speech-recognition-challenge",
        "TensorFlow Speech Recognition Challenge",
        "accuracy",
        True,
        "Audio Classification",
    ),
    (
        "tensorflow2-question-answering",
        "TensorFlow 2.0 Question Answering",
        "micro-f1-score",
        True,
        "Text (Other)",
    ),
    (
        "tgs-salt-identification-challenge",
        "TGS Salt Identification Challenge",
        "mean-precision-intersection-over-union-at-different-thresholds",
        True,
        "Image Segmentation",
    ),
    (
        "tweet-sentiment-extraction",
        "Tweet Sentiment Extraction",
        "jaccard-similarity",
        True,
        "Text Classification",
    ),
    (
        "us-patent-phrase-to-phrase-matching",
        "U.S. Patent Phrase to Phrase Matching",
        "pearson-correlation-coefficient",
        True,
        "Text (Other)",
    ),
    (
        "uw-madison-gi-tract-image-segmentation",
        "UW-Madison GI Tract Image Segmentation",
        "dice-hausdorff-combo",
        True,
        "Image Segmentation",
    ),
    (
        "ventilator-pressure-prediction",
        "Google Brain - Ventilator Pressure Prediction",
        "dice-hausdorff-combo",
        False,
        "Forecasting",
    ),
    (
        "whale-categorization-playground",
        "Humpback Whale Identification Challenge",
        "MAP@5",
        True,
        "Image Classification",
    ),
)

MLEBENCH_MEDIUM = tuple(
    MLETaskSpec(
        experiment_id=f"MBM-{index:02d}",
        task_name=f"mbm_{index:02d}",
        suite="mlebench_medium",
        slug=slug,
        title=title,
        metric=metric,
        higher_is_better=higher_is_better,
        category=category,
        data_source="mlebench",
        feedback_mode="blind",
        submission_limit=None,
    )
    for index, (slug, title, metric, higher_is_better, category) in enumerate(_MBM_ROWS, start=1)
)

_MBH_ROWS = (
    (
        "3d-object-detection-for-autonomous-vehicles",
        "Lyft 3D Object Detection for Autonomous Vehicles",
        "mean-average-precision",
        True,
        "Image Segmentation",
    ),
    (
        "bms-molecular-translation",
        "Bristol-Myers Squibb - Molecular Translation",
        "levenshtein-distance",
        False,
        "Image to Text",
    ),
    (
        "google-research-identify-contrails-reduce-global-warming",
        "Google Research - Identify Contrails to Reduce Global Warming",
        "global-dice",
        True,
        "Image Segmentation",
    ),
    (
        "hms-harmful-brain-activity-classification",
        "HMS - Harmful Brain Activity Classification",
        "KL-divergence",
        False,
        "Image Classification",
    ),
    (
        "iwildcam-2019-fgvc6",
        "iWildCam 2019 - FGVC6",
        "macro-f1-score",
        True,
        "Image Classification",
    ),
    (
        "nfl-player-contact-detection",
        "1st and Future - Player Contact Detection",
        "matthews-correlation-coefficient",
        True,
        "Video Classification",
    ),
    (
        "predict-volcanic-eruptions-ingv-oe",
        "INGV - Volcanic Eruption Prediction",
        "mean-absolute-error",
        False,
        "Signal Processing",
    ),
    (
        "rsna-2022-cervical-spine-fracture-detection",
        "RSNA 2022 Cervical Spine Fracture Detection",
        "weighted-multi-label-log-loss",
        False,
        "Image Classification",
    ),
    (
        "rsna-breast-cancer-detection",
        "RSNA Screening Mammography Breast Cancer Detection",
        "probabilistic-f1-score",
        True,
        "Image Classification",
    ),
    (
        "rsna-miccai-brain-tumor-radiogenomic-classification",
        "RSNA-MICCAI Brain Tumor Radiogenomic Classification",
        "auc-roc",
        True,
        "Image (Other)",
    ),
    (
        "siim-covid19-detection",
        "SIIM-FISABIO-RSNA COVID-19 Detection",
        "mean-average-precision",
        True,
        "Object Detection",
    ),
    (
        "smartphone-decimeter-2022",
        "Google Smartphone Decimeter Challenge 2022",
        "average-haversine-distance",
        False,
        "Tabular",
    ),
    (
        "stanford-covid-vaccine",
        "OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction",
        "multi-class-log-loss",
        False,
        "Tabular",
    ),
    (
        "vesuvius-challenge-ink-detection",
        "Vesuvius Challenge - Ink Detection",
        "f0.5-score",
        True,
        "Image to Image",
    ),
    (
        "vinbigdata-chest-xray-abnormalities-detection",
        "VinBigData Chest X-ray Abnormalities Detection",
        "mAP-at-IoU>0.4",
        True,
        "Object Detection",
    ),
)

MLEBENCH_HIGH = tuple(
    MLETaskSpec(
        experiment_id=f"MBH-{index:02d}",
        task_name=f"mbh_{index:02d}",
        suite="mlebench_high",
        slug=slug,
        title=title,
        metric=metric,
        higher_is_better=higher_is_better,
        category=category,
        data_source="mlebench",
        feedback_mode="blind",
        submission_limit=None,
    )
    for index, (slug, title, metric, higher_is_better, category) in enumerate(_MBH_ROWS, start=1)
)

MLEBENCH_FULL = MLEBENCH_LITE + MLEBENCH_MEDIUM + MLEBENCH_HIGH

_MDU_ROWS = (
    (
        "conways-reverse-game-of-life-2020",
        "Conway's Reverse Game of Life 2020",
        "mean-absolute-error",
        False,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "demand-forecasting-kernels-only",
        "Store Item Demand Forecasting",
        "smape",
        False,
        "Tabular",
        "dojo_dsbench",
    ),
    ("dont-overfit-ii", "Don't Overfit II", "auc-roc", True, "Tabular", "dojo_dsbench"),
    (
        "instant-gratification",
        "Instant Gratification",
        "auc-roc",
        True,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "liverpool-ion-switching",
        "Liverpool Ion Switching",
        "macro-f1",
        True,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "porto-seguro-safe-driver-prediction",
        "Porto Seguro Safe Driver Prediction",
        "normalized-gini",
        True,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "santander-customer-satisfaction",
        "Santander Customer Satisfaction",
        "auc-roc",
        True,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "santander-customer-transaction-prediction",
        "Santander Customer Transaction Prediction",
        "auc-roc",
        True,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "santander-value-prediction-challenge",
        "Santander Value Prediction",
        "rmsle",
        False,
        "Tabular",
        "dojo_dsbench",
    ),
    (
        "20-newsgroups-ciphertext-challenge",
        "20 Newsgroups Ciphertext Challenge",
        "macro-f1",
        True,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "kaggle-llm-science-exam",
        "Kaggle LLM Science Exam",
        "map-at-3",
        True,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "linking-writing-processes-to-writing-quality",
        "Linking Writing Processes to Writing Quality",
        "rmse",
        False,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "llm-detect-ai-generated-text",
        "Detect AI Generated Text",
        "auc-roc",
        True,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "quora-insincere-questions-classification",
        "Quora Insincere Questions Classification",
        "f1",
        True,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "quora-question-pairs",
        "Quora Question Pairs",
        "log-loss",
        False,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "stumbleupon",
        "StumbleUpon Evergreen Classification",
        "auc-roc",
        True,
        "NLP",
        "dojo_kaggle",
    ),
    (
        "airbus-ship-detection",
        "Airbus Ship Detection",
        "mean-f2",
        True,
        "CV",
        "dojo_kaggle",
    ),
    (
        "bengaliai-cv19",
        "Bengali Grapheme Classification",
        "weighted-macro-recall",
        True,
        "CV",
        "dojo_kaggle",
    ),
    (
        "draper-satellite-image-chronology",
        "Satellite Image Chronology",
        "mean-spearman-correlation",
        True,
        "CV",
        "dojo_kaggle",
    ),
    (
        "facial-keypoints-detection",
        "Facial Keypoints Detection",
        "rmse",
        False,
        "CV",
        "dojo_kaggle",
    ),
)

MLEDOJO_UNIQUE = tuple(
    MLETaskSpec(
        experiment_id=f"MDU-{index:02d}",
        task_name=f"mdu_{index:02d}",
        suite="mledojo_unique",
        slug=slug,
        title=title,
        metric=metric,
        higher_is_better=higher_is_better,
        category=category,
        data_source=data_source,
        feedback_mode="interactive",
        submission_limit=15,
    )
    for index, (
        slug,
        title,
        metric,
        higher_is_better,
        category,
        data_source,
    ) in enumerate(_MDU_ROWS, start=1)
)

LEGACY_TASKS = MLEBENCH_LITE + MLEDOJO_UNIQUE
ALL_TASKS = MLEBENCH_FULL + MLEDOJO_UNIQUE
BY_ID = {spec.experiment_id: spec for spec in ALL_TASKS}
BY_TASK_NAME = {spec.task_name: spec for spec in ALL_TASKS}

_CATALOG_SELECTIONS = {
    "all": ALL_TASKS,
    "both": LEGACY_TASKS,
    "mlebench_full": MLEBENCH_FULL,
    "mlebench_high": MLEBENCH_HIGH,
    "mlebench_lite": MLEBENCH_LITE,
    "mlebench_medium": MLEBENCH_MEDIUM,
    "mledojo_unique": MLEDOJO_UNIQUE,
}
CATALOG_SELECTION_NAMES = tuple(_CATALOG_SELECTIONS)


def select_catalog(name: str) -> tuple[MLETaskSpec, ...]:
    """Returns one frozen task selection used by preparation and audits."""
    try:
        return _CATALOG_SELECTIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown MLE catalog selection: {name}") from error

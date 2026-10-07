from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
import hashlib
import io
import json
import pickle
from pathlib import Path
from pprint import pformat
import sqlite3
import textwrap
from typing import Any
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


AUDIT_DB_FILENAME = "training_audit.db"
LEGACY_AUDIT_LOG_FILENAME = "training_audit_log.jsonl"
AUDIT_EXPORT_FILENAME = "training_audit_export.jsonl"


TOOLTIPS = {
    "workflow_mode": "Choose whether to train a new random forest from a CSV or inspect a saved model artifact.",
    "upload_csv": "Upload a CSV file containing feature columns and one column you want to predict.",
    "upload_model": "Upload a `.pkl` file exported by this app to inspect the forest without retraining it.",
    "trust_model_file": "Pickle files can execute code when loaded. Only load files you created yourself or fully trust.",
    "tree_index": "Select which individual tree from the forest to inspect.",
    "tree_depth_limit": "Choose how many levels of the selected tree to draw. You can go all the way to that tree's full depth and use zoom controls to inspect the result.",
    "date_columns": "Choose any columns that should be parsed as dates before training.",
    "date_dayfirst": "Enable this if your dates are written like DD/MM/YYYY instead of MM/DD/YYYY.",
    "enable_pair_column": "Create a new text column by combining two existing columns row by row.",
    "pair_column_name": "Name of the new paired column that will be added to the dataset.",
    "pair_column_left": "First source column used in the row-by-row pairing.",
    "pair_column_right": "Second source column used in the row-by-row pairing.",
    "pair_column_separator": "Text inserted between the two source values in the new paired column.",
    "enable_group_average": "Create a new numeric column containing the average of one column within each group of another column.",
    "group_average_name": "Name of the new grouped-average column that will be added to the dataset.",
    "group_average_value": "Numeric column whose values will be averaged within each group.",
    "group_average_group": "Column used to define the groups before the average is calculated.",
    "target_column": "The output column the model will learn to predict.",
    "feature_columns": "The input columns used to predict the target. Leave out IDs or columns that leak the answer.",
    "problem_type": "Auto-detect inspects the target column. Choose Classification for categories or Regression for numeric values.",
    "test_size": "Fraction of rows held out for final evaluation. Example: 0.20 means 20% test and 80% train.",
    "random_state": "Seed used for repeatable data splits and repeatable random forest training.",
    "shuffle_rows": "Shuffle the dataset before splitting into train and test sets. Usually this should stay enabled.",
    "stratify_split": "For classification only, keeps class proportions similar in the train and test sets.",
    "run_cv": "Runs cross-validation on the training set for a more stable estimate of model performance.",
    "cv_folds": "Number of folds to use for cross-validation. More folds take longer to run.",
    "n_estimators": "Number of trees in the forest. More trees usually improve stability but increase training time.",
    "class_weight": "Changes how strongly the model treats each class. Useful when one class appears much less often than another.",
    "criterion_classification": "Split quality metric for classification trees.",
    "criterion_regression": "Split quality metric for regression trees.",
    "use_max_depth": "Enable a depth limit for each tree. Limiting depth can reduce overfitting.",
    "max_depth": "Maximum number of levels allowed in each tree when depth limiting is enabled.",
    "min_samples_split": "Minimum number of samples required before an internal node can be split.",
    "min_samples_leaf": "Minimum number of samples that must remain in each leaf node.",
    "max_features": "How many features each split can consider. Smaller values add randomness between trees.",
    "bootstrap": "If enabled, each tree trains on a sampled subset of the training rows with replacement.",
    "use_max_samples": "Enable a custom fraction of training rows for each bootstrapped tree.",
    "max_samples": "Fraction of training rows sampled for each tree when bootstrap is enabled.",
    "train_model": "Train the model using the selected columns and settings.",
}

STORY_SECTIONS = [
    {
        "id": "setup",
        "kicker": "The Setup",
        "title": "A Dangerous Forest",
        "chips": ["1,000 mushrooms", "22 observations", "poison vs edible"],
        "paragraphs": [
            "Imagine you are a forager with 1,000 mushroom records. Some became dinner. Others sent people to the hospital.",
            "Each mushroom has detailed notes: cap color, size, gill color, smell, whether it grows in clusters, and more. The job is to pass that experience on without anyone learning the hard way.",
        ],
        "controls": [
            ("Target column", "This is the poisonous-or-edible outcome the forest is trying to learn."),
            ("Feature columns", "These are the clues in the notebook: smell, color, size, clusters, and the rest."),
            ("Problem type", "For a mushroom story like this, the forest is usually doing classification."),
            ("Columns to convert to dates", "Use this when your notebook includes time-based clues that need to become usable signals."),
        ],
        "visual": [92, 68, 84],
    },
    {
        "id": "overfit",
        "kicker": "The First Problem",
        "title": "The Overconfident Expert",
        "chips": ["one tree", "perfect memory", "bad generalization"],
        "paragraphs": [
            "A single decisive expert can build a rulebook that explains every mushroom you already saw. The trouble starts when that expert mistakes coincidences for truth.",
            "That is overfitting: the model memorizes your old stories instead of learning the pattern that will survive in a new forest.",
        ],
        "controls": [
            ("Set max_depth / max_depth", "Limits how deep one tree can go before it turns into an overconfident storyteller."),
            ("min_samples_split", "Prevents tiny groups from being split into overly specific branches."),
            ("min_samples_leaf", "Forces each ending branch to keep enough evidence behind it."),
        ],
        "visual": [96, 44, 28],
    },
    {
        "id": "crowd",
        "kicker": "The Insight",
        "title": "Wisdom of the Crowd",
        "chips": ["many experts", "different angles", "majority vote"],
        "paragraphs": [
            "Instead of trusting one loud expert, Random Forest creates a crowd of reasonable ones. Each tree gets its own slightly different experience and viewpoint.",
            "One weak tree can be strange. A diverse crowd of trees is much harder to fool.",
        ],
        "controls": [
            ("n_estimators", "This is how many voices join the vote. More trees mean a larger crowd."),
            ("random_state", "Locks the crowd into a repeatable set of remixed stories and feature draws."),
        ],
        "visual": [54, 88, 100],
    },
    {
        "id": "bootstrap",
        "kicker": "Part One",
        "title": "Bootstrap: Learning from Stories",
        "chips": ["sampling with replacement", "remixed training sets", "different memories"],
        "paragraphs": [
            "Each tree learns from a remix of the original mushroom notebook. Some mushrooms appear twice. Some never show up at all.",
            "That remix matters because each tree becomes wrong in its own slightly different way instead of sharing the exact same blind spots.",
        ],
        "controls": [
            ("bootstrap", "Turns the remixed bag of stories on or off."),
            ("Set max_samples / max_samples", "Controls how much of the bag each tree is allowed to see."),
            ("random_state", "Repeats the same remixed bag when you want stable experiments."),
        ],
        "visual": [80, 62, 74],
    },
    {
        "id": "random_features",
        "kicker": "Part Two",
        "title": "Random Features: Forcing Different Perspectives",
        "chips": ["blindfolded experts", "different first questions", "decorrelated trees"],
        "paragraphs": [
            "If every tree always starts with the strongest clue, they all begin to look the same. Random Forest avoids that by restricting which clues are available at each split.",
            "One tree might focus on smell. Another starts from cap color. Another leans on gill shape. The disagreement is a feature, not a bug.",
        ],
        "controls": [
            ("max_features", "This is the blindfold. It limits how many clues a tree can consider at each split."),
            ("criterion", "This controls how a tree judges whether one question creates a cleaner split than another."),
        ],
        "visual": [38, 76, 92],
    },
    {
        "id": "growing",
        "kicker": "Part Three",
        "title": "Growing the Forest",
        "chips": ["many trees", "controlled complexity", "shared stability"],
        "paragraphs": [
            "Each student now grows their own decision tree. Some grow deeper. Some stop early. Some are shaped by smell patterns, others by visual cues.",
            "The overall forest becomes strong because each tree stays useful without being allowed to become all-knowing.",
        ],
        "controls": [
            ("n_estimators", "Controls how many individual trees the forest grows."),
            ("max_depth", "Caps how elaborate each tree can become."),
            ("min_samples_split", "Sets how much evidence is needed before the tree keeps branching."),
            ("min_samples_leaf", "Sets how much evidence must remain at the leaves."),
        ],
        "visual": [70, 84, 58],
    },
    {
        "id": "prediction",
        "kicker": "The Prediction",
        "title": "Taking a Vote",
        "chips": ["new mushroom", "100 opinions", "confidence from agreement"],
        "paragraphs": [
            "When a new mushroom arrives, the forest asks every tree for an answer. Some say poisonous. Some say edible. The final prediction comes from the collective vote.",
            "What makes the vote powerful is not just the winner. It is the shape of the disagreement and which clues drove it.",
        ],
        "controls": [
            ("class_weight", "If one class is rarer or more costly, this tells the forest to take that side more seriously."),
            ("Train Model", "This triggers the full journey from many individual tree opinions to one final decision."),
        ],
        "visual": [87, 13, 76],
    },
    {
        "id": "reality_check",
        "kicker": "The Hidden Gift",
        "title": "Out-of-Bag Error and Other Reality Checks",
        "chips": ["unseen examples", "honest evaluation", "trust but verify"],
        "paragraphs": [
            "Because every tree misses part of the training data, a forest can often judge itself by asking only the trees that never saw a given example. That built-in check is the OOB idea.",
            "This app does not expose OOB directly yet, so the closest equivalents are the held-out test split and optional cross-validation.",
        ],
        "controls": [
            ("Test size", "Sets aside a slice of the data as mushrooms the forest never gets to study during fitting."),
            ("Run cross-validation", "Repeats the reality check across multiple train/test rotations."),
            ("CV folds", "Controls how many rotations you run for that repeated reality check."),
        ],
        "visual": [44, 72, 66],
    },
    {
        "id": "importance",
        "kicker": "What Actually Matters",
        "title": "Feature Importance",
        "chips": ["strong clues", "weak clues", "pattern tracing"],
        "paragraphs": [
            "After the voting ends, you can look back and ask which clues actually changed decisions across the whole forest. Smell might repeatedly create clean splits. Cap color might be noisier.",
            "That retrospective view is one of the forest's best gifts: it tells you which observations kept reducing uncertainty.",
        ],
        "controls": [
            ("Feature columns", "Choosing better clues changes which signals the forest can rely on."),
            ("Columns to convert to dates", "Date-derived parts can become important once they are expanded into usable features."),
            ("Top feature importances", "This results table is where the story reveals which clues mattered most."),
        ],
        "visual": [91, 52, 34],
    },
    {
        "id": "errors",
        "kicker": "Why This Works",
        "title": "The Story of Errors",
        "chips": ["bias", "variance", "stability through averaging"],
        "paragraphs": [
            "Random Forest mainly fights variance. It takes many slightly wrong perspectives and averages away their private obsessions.",
            "It does not magically fix bad source data, but it does make the model steadier, less brittle, and less likely to panic over one odd memory.",
        ],
        "controls": [
            ("n_estimators", "More trees help random quirks cancel out."),
            ("max_features", "More randomness in each split makes the trees less likely to share the same mistake."),
            ("max_depth / min_samples_split / min_samples_leaf", "These keep individual trees humble enough to generalize."),
        ],
        "visual": [64, 48, 90],
    },
    {
        "id": "scenario",
        "kicker": "The Forest in Action",
        "title": "One Mushroom, Many Opinions",
        "chips": ["red cap", "white gills", "almond smell"],
        "paragraphs": [
            "A new mushroom arrives: red cap, white gills, almond smell, medium size, grows alone. One tree loves the almond smell and says edible. Another fixates on the red cap and white gills and says poisonous.",
            "The forest aggregates those disagreements and produces a decision that is more cautious than any single tree on its own.",
        ],
        "controls": [
            ("Predictions sample", "Use this table to inspect where the forest agrees with reality and where it hesitates."),
            ("Confusion matrix", "For classification, this shows which types of mushrooms the forest mixes up."),
            ("Feature importances", "This helps explain why one clue won the argument over another."),
        ],
        "visual": [72, 41, 88],
    },
    {
        "id": "ending",
        "kicker": "The End of the Story",
        "title": "Diversity, Democracy, Humility",
        "chips": ["diversity", "democracy", "humility"],
        "paragraphs": [
            "Random Forest is not one genius. It is a hundred reasonable people, each with limited experience, forced to look at the problem differently and then vote.",
            "Its power comes from diversity in data, democracy in prediction, and humility in how much any single tree is allowed to claim.",
        ],
        "controls": [
            ("All training controls", "Each setting adjusts how diverse, how democratic, or how humble the forest becomes."),
            ("Download trained model", "This lets you keep the forest after its many voices have settled on a working pattern."),
        ],
        "visual": [100, 78, 58],
    },
]

STORY_TOOLTIP_LINKS = {
    "workflow_mode": "The End of the Story. This lets you either grow a new forest or revisit one you already trained.",
    "upload_csv": "The Setup. This is the notebook of mushroom stories the forest learns from.",
    "upload_model": "The End of the Story. This is how you return to an already-grown forest and study it again.",
    "trust_model_file": "The End of the Story. You should only reopen forests whose origin you trust.",
    "tree_index": "The Forest in Action. This lets you interview one student in the crowd at a time.",
    "tree_depth_limit": "The Overconfident Expert. This keeps the single-tree view readable instead of showing every last branch at once.",
    "date_columns": "The Setup and Feature Importance. These are extra clues you want the forest to interpret correctly before the vote starts.",
    "date_dayfirst": "The Setup. It changes how the forest reads the field notes before training begins.",
    "enable_pair_column": "The Setup and Feature Importance. This lets you turn two separate clues into one combined clue before the forest starts learning.",
    "pair_column_name": "The Setup. This is the label for the new combined clue in the notebook.",
    "pair_column_left": "The Setup. This picks the first clue in the combined pair.",
    "pair_column_right": "The Setup. This picks the second clue in the combined pair.",
    "pair_column_separator": "The Setup. This controls how the combined clue is written in the notebook.",
    "enable_group_average": "Feature Importance. This creates a summary clue by averaging one measurement inside each group.",
    "group_average_name": "Feature Importance. This is the label for the new summary clue in the notebook.",
    "group_average_value": "Feature Importance. This chooses the numeric measurement you want to average.",
    "group_average_group": "Feature Importance. This chooses which clue defines the groups before averaging.",
    "target_column": "The Setup. This is the poisonous-or-edible outcome the students are trying to predict.",
    "feature_columns": "The Setup and Feature Importance. These are the clues each tree can question along the way.",
    "problem_type": "The Setup and Taking a Vote. It determines whether the forest is choosing categories or predicting a timeline.",
    "test_size": "Out-of-Bag Error and other reality checks. This is your held-out patch of forest for honest evaluation.",
    "random_state": "Wisdom of the Crowd and Bootstrap. It keeps the same crowd and same remixed stories when you rerun training.",
    "shuffle_rows": "Bootstrap and reality checks. It mixes the mushroom notebook before the split so patterns are not grouped accidentally.",
    "stratify_split": "Taking a Vote. It keeps class balance steady so both poisonous and edible voices stay represented in train and test.",
    "run_cv": "Out-of-Bag Error and other reality checks. This is the closest in-app version of asking many unseen mini-forests to vote.",
    "cv_folds": "Out-of-Bag Error and other reality checks. More folds mean more repeated honesty checks.",
    "n_estimators": "Wisdom of the Crowd and Growing the Forest. This is the size of the voting crowd.",
    "class_weight": "Taking a Vote. It tells the forest to listen harder when one class is rarer or riskier.",
    "criterion_classification": "Random Features. It defines what a tree counts as a good question when separating edible from poisonous.",
    "criterion_regression": "Random Features. It defines what a tree counts as a good question when predicting a numeric timeline.",
    "use_max_depth": "The Overconfident Expert. Turn this on when you want to keep each tree from becoming too sure of itself.",
    "max_depth": "The Overconfident Expert and Growing the Forest. This is how far one student's rulebook is allowed to expand.",
    "min_samples_split": "The Overconfident Expert. It stops a tree from carving tiny groups into overly specific stories.",
    "min_samples_leaf": "The Overconfident Expert. It forces every leaf to keep enough evidence behind its claim.",
    "max_features": "Random Features: Forcing Different Perspectives. This is the blindfold that makes trees ask different first questions.",
    "bootstrap": "Bootstrap: Learning from Stories. This turns the remixed bag of mushroom experiences on or off.",
    "use_max_samples": "Bootstrap: Learning from Stories. This lets you decide how much of the story bag each tree can draw from.",
    "max_samples": "Bootstrap: Learning from Stories. This is the fraction of examples each tree receives from the bag.",
    "train_model": "The Forest in Action. This is the moment the crowd goes from stories to a final vote.",
}

DEFAULT_TOOLTIP_SELECTION_TIP = (
    "Start with the default, change one setting at a time, and keep the change only "
    "if held-out results improve."
)

TOOLTIP_SELECTION_TIPS = {
    "workflow_mode": (
        "Choose Train for new CSV experiments. Choose Inspect only when you already "
        "have a saved `.pkl` and just want to study it."
    ),
    "upload_csv": (
        "Use the cleanest representative dataset you have. Include the target column "
        "and only rows you would trust in real use."
    ),
    "upload_model": (
        "Use a model file exported by this app for the same kind of problem you want "
        "to inspect."
    ),
    "trust_model_file": (
        "Only enable this for files you created or can verify. An unknown pickle "
        "should never be treated as safe."
    ),
    "tree_index": (
        "Start with a typical-looking tree, then inspect a few others to see whether "
        "important splits repeat across the forest."
    ),
    "tree_depth_limit": (
        "Begin around 3 to 6 for readability and only raise it when you need to trace "
        "deeper decision paths."
    ),
    "date_columns": (
        "Select only columns that are truly dates and could carry timing signal such "
        "as seasonality, recency, or weekday effects."
    ),
    "date_dayfirst": (
        "Turn this on only if your source dates are mostly written as DD/MM/YYYY or a "
        "similar day-first format."
    ),
    "enable_pair_column": (
        "Use this when two text columns together carry more meaning than either one on "
        "its own."
    ),
    "pair_column_name": (
        "Give it a descriptive name that explains the combined meaning, not just the "
        "mechanics of the join."
    ),
    "pair_column_left": (
        "Pick the first field that contributes meaning to the combined text feature."
    ),
    "pair_column_right": (
        "Pick the second field that adds context the first field misses on its own."
    ),
    "pair_column_separator": (
        "Use a separator that keeps the combined value readable without appearing "
        "inside the source values too often."
    ),
    "enable_group_average": (
        "Use this when a category-level average could be predictive and you have "
        "enough rows per group to make that average stable."
    ),
    "group_average_name": (
        "Name it after the pattern it represents, such as `avg_price_by_store`."
    ),
    "group_average_value": (
        "Choose a numeric column with stable meaning. Noisy or sparse measures rarely "
        "help much."
    ),
    "group_average_group": (
        "Choose a grouping column whose categories repeat often enough to form "
        "reliable averages."
    ),
    "target_column": (
        "Pick the outcome you want to predict. It should not also be used directly or "
        "indirectly as a feature."
    ),
    "feature_columns": (
        "Start with inputs known before the outcome happens. Remove IDs, leaked "
        "answers, and mostly empty fields."
    ),
    "problem_type": (
        "Use Classification for labels or categories. Use Regression for continuous "
        "numeric values or dates converted to a timeline."
    ),
    "test_size": (
        "Start around 0.20. Increase it if you have plenty of data and want a stricter "
        "final check."
    ),
    "random_state": (
        "Keep one fixed value while comparing settings so differences come from the "
        "model, not from a new random split."
    ),
    "shuffle_rows": (
        "Leave this on unless row order is meaningful time order that should stay "
        "intact."
    ),
    "stratify_split": (
        "Enable this for imbalanced classification so rare classes stay represented in "
        "both train and test."
    ),
    "run_cv": (
        "Turn this on when the dataset is modest or results bounce around. Skip it on "
        "very large data if speed matters more."
    ),
    "cv_folds": (
        "Use 5 as a default. Drop to 3 for speed or raise toward 10 only when the "
        "dataset is large enough."
    ),
    "n_estimators": (
        "Start around 200. Increase until your metrics stabilize, then stop when extra "
        "trees bring little improvement."
    ),
    "class_weight": (
        "Use `balanced` when classes are uneven or mistakes on the rarer class matter "
        "more."
    ),
    "criterion_classification": (
        "Start with `gini`. Try `entropy` or `log_loss` only if validation metrics "
        "improve."
    ),
    "criterion_regression": (
        "Start with `squared_error`. Use `absolute_error` for more outlier-robust "
        "behavior and `poisson` only for non-negative count-like targets."
    ),
    "use_max_depth": (
        "Turn this on when trees seem to overfit, training is slow, or you want "
        "simpler tree behavior."
    ),
    "max_depth": (
        "Start around 8 to 15 on mixed tabular data. Lower it if train results are far "
        "better than test results."
    ),
    "min_samples_split": (
        "Increase this when trees are fragmenting into tiny groups. `2` is the normal "
        "baseline."
    ),
    "min_samples_leaf": (
        "Try `1` first, then raise it to 2 to 10 if predictions look noisy or too "
        "sensitive to small data changes."
    ),
    "max_features": (
        "Start with `sqrt` unless you have a strong reason not to. Use more features "
        "only if extra randomness hurts validation performance."
    ),
    "bootstrap": (
        "Leave this on for a classic random forest. Turn it off only if validation "
        "results clearly improve without bagging."
    ),
    "use_max_samples": (
        "Turn this on when you want more diversity between trees or faster training on "
        "large datasets."
    ),
    "max_samples": (
        "Start around 0.60 to 0.90 when bootstrapping. Lower values add diversity but "
        "can make each tree weaker."
    ),
    "train_model": (
        "Run a baseline first, inspect the metrics, then change one setting at a time "
        "so you can see what actually helped."
    ),
}

STORY_DOCUMENT_NAME = "ForestStory.docx"

STORY_CONTROL_SELECTION_TIPS = {
    "Target column": (
        "Choose the real outcome you want to predict, and keep anything derived from "
        "that outcome out of the feature set."
    ),
    "Feature columns": (
        "Keep columns that are known before the outcome and likely to contain signal. "
        "Drop IDs, leakage, and fields with little usable information."
    ),
    "Problem type": (
        "Use classification for categories and regression for continuous numbers or "
        "date values represented on a timeline."
    ),
    "Columns to convert to dates": (
        "Convert only true date or timestamp fields, especially when timing patterns "
        "like month, weekday, or recency might matter."
    ),
    "Set max_depth / max_depth": (
        "Turn this on when trees are memorizing too much detail. A starting range of 8 "
        "to 15 is usually a good sanity check."
    ),
    "min_samples_split": (
        "Raise this when the forest keeps carving tiny groups into separate branches. "
        "Leave it low if you need finer splits."
    ),
    "min_samples_leaf": (
        "Raise this when predictions look noisy or unstable. Higher values force each "
        "leaf to keep more evidence behind it."
    ),
    "n_estimators": (
        "Start around 200 and increase until validation metrics stop moving in a "
        "meaningful way."
    ),
    "random_state": (
        "Keep one seed fixed while tuning so you can compare settings without adding "
        "extra randomness."
    ),
    "bootstrap": (
        "Leave it on unless you test otherwise and see a clear improvement. It is part "
        "of the standard random-forest recipe."
    ),
    "Set max_samples / max_samples": (
        "If bootstrap is on, start around 0.70 to 0.90 when you want more tree "
        "diversity. Leave it unset if the default bag size already works well."
    ),
    "max_features": (
        "Start with `sqrt` and only widen the feature pool if the added randomness is "
        "hurting validation performance."
    ),
    "criterion": (
        "Begin with the default for your problem type and switch only when cross-"
        "validation or the held-out split shows a real gain."
    ),
    "max_depth": (
        "Use a lower depth when training performance is much stronger than test "
        "performance. Use a higher one only if the forest is clearly underfitting."
    ),
    "class_weight": (
        "Reach for this when one class is rare or its mistakes are more expensive than "
        "others."
    ),
    "Train Model": (
        "Train a sensible baseline first, then compare controlled changes one at a "
        "time so you know which setting actually helped."
    ),
    "Predictions sample": (
        "Look for repeated failure patterns rather than isolated misses. Those patterns "
        "tell you which features or constraints may need work."
    ),
    "Test size": (
        "About 20% is a solid default. Use more only if you have enough data to spare "
        "for a tougher final check."
    ),
    "Run cross-validation": (
        "Turn it on when one train/test split feels noisy or the dataset is not very "
        "large."
    ),
    "CV folds": (
        "Use 5 by default. Lower it for speed, or raise it only when you have enough "
        "rows for each fold to stay representative."
    ),
    "Top feature importances": (
        "Treat this as directional, not absolute. Use it to spot weak, suspicious, or "
        "surprisingly strong signals."
    ),
    "max_depth / min_samples_split / min_samples_leaf": (
        "Tune these together when the forest overfits. Lower depth or raise sample "
        "thresholds before adding more complexity elsewhere."
    ),
    "Confusion matrix": (
        "Focus on which classes the model confuses most often. That usually points to "
        "class weighting, feature gaps, or threshold issues."
    ),
    "Feature importances": (
        "Use them to see which signals the model relies on most and to question any "
        "importance that looks like leakage."
    ),
    "All training controls": (
        "Start with the defaults, set a baseline, then tune only the settings that "
        "target your current problem: overfitting, instability, imbalance, or speed."
    ),
    "Download trained model": (
        "Save the model once the metrics and settings are stable enough that you would "
        "want to reuse that exact version."
    ),
}

STORY_SECTION_HELPERS = {
    "The Setup: A Dangerous Forest": [
        ("Target column", "This is the poisonous-or-edible truth the forest is trying to learn."),
        ("Feature columns", "These are the clues in the notebook: smell, cap color, gills, size, and more."),
        ("Problem type", "For this mushroom story, the forest is usually solving a classification problem."),
        ("Columns to convert to dates", "Use this when a clue in the notebook is time-based and needs to become a usable feature."),
    ],
    "The First Problem: The Overconfident Expert": [
        ("Set max_depth / max_depth", "These keep one tree from becoming too confident and too specific."),
        ("min_samples_split", "This stops a tree from inventing fragile branches from tiny groups."),
        ("min_samples_leaf", "This forces each leaf to keep enough evidence behind its claim."),
    ],
    "The Insight: Wisdom of the Crowd": [
        ("n_estimators", "This is the size of the crowd that gets to vote."),
        ("random_state", "This keeps the same crowd and same random remix when you rerun the experiment."),
    ],
    "Part One: The Bootstrap—Learning from Stories": [
        ("bootstrap", "Turns the remixed bag of mushroom stories on or off."),
        ("Set max_samples / max_samples", "Controls how much of the remixed bag each tree receives."),
        ("random_state", "Reuses the same bag remix when you want repeatable runs."),
    ],
    "Part Two: Random Features—Forcing Different Perspectives": [
        ("max_features", "This is the blindfold that limits which clues each split may consider."),
        ("criterion", "This decides how a tree judges whether one question is cleaner than another."),
    ],
    "Part Three: Growing the Forest": [
        ("n_estimators", "Controls how many trees the forest grows."),
        ("max_depth", "Caps how intricate one tree is allowed to become."),
        ("min_samples_split", "Controls when a branch is allowed to keep splitting."),
        ("min_samples_leaf", "Controls how much evidence must remain at the leaves."),
    ],
    "The Prediction: Taking a Vote": [
        ("class_weight", "Tells the forest to listen more carefully when one class is rarer or riskier."),
        ("Train Model", "This starts the full learning-and-voting process."),
        ("Predictions sample", "This is where you inspect the forest's final answer on held-out examples."),
    ],
    "The Hidden Gift: Out-of-Bag Error": [
        ("Test size", "This is the app's held-out patch of forest for honest evaluation."),
        ("Run cross-validation", "This is the closest in-app equivalent to repeated OOB-style reality checks."),
        ("CV folds", "Controls how many times that reality check repeats."),
    ],
    "Feature Importance: What Actually Matters?": [
        ("Feature columns", "Choosing better clues changes what the forest can rely on."),
        ("Columns to convert to dates", "Date-derived clues can become important once expanded into usable parts."),
        ("Top feature importances", "This results table shows which clues shaped the forest's decisions most."),
    ],
    "Why This Works: The Story of Errors": [
        ("n_estimators", "More trees help random quirks cancel out."),
        ("max_features", "More randomized perspectives reduce shared mistakes."),
        ("max_depth / min_samples_split / min_samples_leaf", "These keep each tree humble enough to generalize."),
    ],
    "The Forest in Action: A Real Scenario": [
        ("Predictions sample", "Use this to inspect where the forest agreed with reality and where it hesitated."),
        ("Confusion matrix", "For classification, this shows which mushroom types the forest mixes up."),
        ("Feature importances", "This helps explain why one clue won the argument over another."),
    ],
    "The End of the Story": [
        ("All training controls", "Together they shape how diverse, democratic, and humble the forest becomes."),
        ("Download trained model", "This lets you keep the forest after the vote is over."),
    ],
}

NANOSECONDS_PER_DAY = 86_400_000_000_000

BASE_THEME_CSS = """
<style>
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stHeader"] button[kind="header"],
header button[kind="header"] {
    display: none !important;
}

[data-testid="stHeader"],
header {
    background: transparent !important;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 21rem !important;
    max-width: 21rem !important;
    width: 21rem !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
}

[data-testid="stSidebar"][aria-expanded="false"] > div {
    width: 21rem !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

.st-key-forest_toggle_shell {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    z-index: 1000;
    width: auto;
}

.st-key-story_toggle_shell {
    position: fixed;
    left: 1rem;
    bottom: 1rem;
    z-index: 1000;
    width: auto;
}

.st-key-forest_toggle_shell > div {
    margin: 0;
}

.st-key-story_toggle_shell > div {
    margin: 0;
}

.st-key-forest_toggle_shell button,
.st-key-story_toggle_shell button {
    min-height: 2.9rem;
    padding: 0 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(18, 24, 20, 0.12);
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 14px 30px rgba(12, 18, 14, 0.14);
    backdrop-filter: blur(10px);
}

.st-key-forest_toggle_shell button p,
.st-key-story_toggle_shell button p {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.tree-rules {
    margin-top: 0.5rem;
}

.tree-rules details {
    margin: 0.45rem 0;
    padding: 0.45rem 0 0.45rem 0.75rem;
    border-left: 2px solid rgba(30, 38, 32, 0.14);
}

.tree-rules summary {
    cursor: pointer;
    font-weight: 600;
}

.tree-rules .tree-rules-note,
.tree-rules .tree-rules-question {
    margin: 0.35rem 0 0.5rem;
    line-height: 1.55;
}

.tree-rules .tree-rules-body {
    margin-top: 0.35rem;
}

.tree-rules .tree-leaf {
    margin-top: 0.4rem;
    line-height: 1.55;
    white-space: pre-wrap;
}
</style>
"""

FOREST_THEME_CSS = """
<style>
html, body {
    color-scheme: dark;
}

.stApp {
    background:
        radial-gradient(circle at 50% 0%, rgba(229, 240, 232, 0.08), transparent 30%),
        linear-gradient(180deg, #06100b 0%, #0a130e 24%, #0d1611 100%);
    color: #edf5ee;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container,
[data-testid="stSidebar"] > div {
    position: relative;
    z-index: 2;
}

.block-container {
    max-width: 1120px;
    padding-top: 1rem;
}

h1, h2, h3 {
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
    letter-spacing: 0.03em;
    color: #f1f6f2;
}

body, p, label, li, span, input, textarea {
    font-family: "Trebuchet MS", "Lucida Sans Unicode", sans-serif;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #dbe7dd;
}

a {
    color: #f1f6f2 !important;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(7, 11, 9, 0.94) 0%, rgba(13, 19, 15, 0.9) 100%);
    border-right: 1px solid rgba(220, 235, 223, 0.12);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] * {
    color: #e7f0e8;
}

[data-testid="stFileUploaderDropzone"],
div[data-testid="metric-container"],
[data-testid="stExpander"],
div[data-testid="stDataFrame"],
.stAlert {
    background: rgba(9, 15, 11, 0.62) !important;
    border: 1px solid rgba(232, 242, 234, 0.11) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
    backdrop-filter: blur(12px);
}

[data-testid="stFileUploaderDropzone"] {
    border-style: solid !important;
}

[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] small {
    color: #dfe9e1 !important;
    fill: #dfe9e1 !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
textarea {
    background: rgba(12, 18, 14, 0.78) !important;
    border: 1px solid rgba(235, 245, 237, 0.16) !important;
    border-radius: 14px !important;
    color: #f1f6f2 !important;
}

div[data-baseweb="tag"] {
    background: rgba(196, 214, 199, 0.14) !important;
    border: 1px solid rgba(228, 240, 230, 0.14) !important;
}

button[kind="primary"],
button[kind="secondary"],
button[kind="tertiary"],
.stDownloadButton button {
    border-radius: 16px !important;
    border: 1px solid rgba(233, 241, 234, 0.16) !important;
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
}

button[kind="primary"] {
    background: linear-gradient(180deg, rgba(233, 241, 234, 0.16), rgba(191, 205, 194, 0.08)) !important;
}

button[kind="secondary"],
button[kind="tertiary"],
.stDownloadButton button {
    background: rgba(14, 20, 15, 0.72) !important;
}

button[kind="primary"] p,
button[kind="secondary"] p,
button[kind="tertiary"] p,
.stDownloadButton button p {
    color: #edf5ee !important;
    letter-spacing: 0.04em;
}

div[data-testid="stMetricLabel"] p {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
}

[data-testid="stTooltipIcon"] button {
    background: rgba(236, 244, 238, 0.12) !important;
    border: 1px solid rgba(236, 244, 238, 0.28) !important;
    border-radius: 999px !important;
    color: #f5fbf6 !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.22);
}

[data-testid="stTooltipIcon"] button:hover {
    background: rgba(236, 244, 238, 0.18) !important;
    border-color: rgba(244, 250, 245, 0.42) !important;
}

[data-testid="stTooltipIcon"] button svg {
    fill: #f5fbf6 !important;
    color: #f5fbf6 !important;
    stroke: #f5fbf6 !important;
}

[data-baseweb="tooltip"] {
    background: rgba(247, 251, 248, 0.98) !important;
    color: #0c1510 !important;
    border: 1px solid rgba(26, 42, 31, 0.16) !important;
    box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28) !important;
    max-width: 32rem !important;
    line-height: 1.5 !important;
    white-space: pre-line !important;
}

[data-baseweb="tooltip"] * {
    color: #0c1510 !important;
    fill: #0c1510 !important;
    stroke: transparent;
}

.forest-backdrop {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

.forest-backdrop .forest-haze {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 50% 18%, rgba(231, 245, 233, 0.14), transparent 24%),
        radial-gradient(circle at 50% 32%, rgba(174, 194, 178, 0.08), transparent 28%),
        linear-gradient(180deg, rgba(8, 12, 10, 0.08), rgba(8, 12, 10, 0.0) 30%, rgba(7, 10, 8, 0.32) 100%);
}

.forest-backdrop .forest-facets-top {
    position: absolute;
    inset: 0 0 auto 0;
    height: 42vh;
    background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 560'><polygon fill='%230d1712' points='0,360 130,210 280,340 420,170 575,300 720,145 920,315 1090,120 1260,275 1420,100 1600,250 1600,0 0,0'/><polygon fill='%2316251d' points='0,380 155,245 305,360 445,205 590,320 770,195 955,345 1140,175 1335,305 1600,175 1600,0 0,0'/><polygon fill='%231d3025' points='0,415 190,285 350,400 505,250 675,380 860,235 1040,395 1220,225 1425,360 1600,225 1600,0 0,0'/></svg>") center top / cover no-repeat;
    opacity: 0.52;
}

.forest-backdrop .forest-ridge-back {
    position: absolute;
    left: -5vw;
    bottom: 24vh;
    width: 110vw;
    height: 28vh;
    background: linear-gradient(135deg, rgba(28, 43, 33, 0.4) 0 18%, rgba(17, 28, 21, 0.45) 18% 39%, rgba(42, 62, 48, 0.5) 39% 58%, rgba(15, 24, 18, 0.42) 58% 100%);
    clip-path: polygon(0 100%, 0 70%, 8% 52%, 16% 62%, 23% 40%, 31% 58%, 40% 26%, 48% 44%, 56% 19%, 64% 43%, 73% 15%, 82% 37%, 90% 24%, 100% 50%, 100% 100%);
    opacity: 0.44;
}

.forest-backdrop .forest-ridge-mid {
    position: absolute;
    left: -6vw;
    bottom: 12vh;
    width: 112vw;
    height: 35vh;
    background: linear-gradient(135deg, rgba(27, 43, 31, 0.58) 0 22%, rgba(14, 22, 17, 0.66) 22% 43%, rgba(33, 52, 38, 0.62) 43% 61%, rgba(18, 29, 21, 0.68) 61% 100%);
    clip-path: polygon(0 100%, 0 74%, 6% 56%, 13% 66%, 20% 48%, 28% 60%, 37% 34%, 46% 51%, 55% 27%, 64% 47%, 72% 20%, 80% 42%, 89% 24%, 100% 46%, 100% 100%);
}

.forest-backdrop .forest-ridge-front {
    position: absolute;
    left: -4vw;
    bottom: -4vh;
    width: 108vw;
    height: 40vh;
    background: linear-gradient(135deg, rgba(18, 29, 21, 0.92) 0 18%, rgba(10, 17, 13, 0.95) 18% 39%, rgba(28, 43, 31, 0.95) 39% 58%, rgba(13, 21, 16, 0.97) 58% 100%);
    clip-path: polygon(0 100%, 0 74%, 7% 61%, 14% 72%, 23% 50%, 31% 65%, 39% 39%, 48% 57%, 58% 30%, 66% 52%, 75% 22%, 83% 44%, 92% 28%, 100% 40%, 100% 100%);
    box-shadow: inset 0 40px 120px rgba(0, 0, 0, 0.22);
}

.forest-backdrop .forest-trees-back {
    position: absolute;
    left: 0;
    bottom: 12vh;
    width: 100%;
    height: 24vh;
    background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 960 220'><polygon fill='%23111b15' points='18,220 66,110 114,220'/><polygon fill='%2315241b' points='84,220 148,74 212,220'/><polygon fill='%23111b15' points='182,220 246,98 310,220'/><polygon fill='%2318281e' points='292,220 370,52 448,220'/><polygon fill='%23111b15' points='430,220 490,104 550,220'/><polygon fill='%231a2c21' points='530,220 612,64 694,220'/><polygon fill='%23111b15' points='664,220 726,118 788,220'/><polygon fill='%2315241b' points='760,220 840,70 920,220'/></svg>") center bottom / 700px 180px repeat-x;
    opacity: 0.44;
}

.forest-backdrop .forest-trees-front {
    position: absolute;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 30vh;
    background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 960 240'><polygon fill='%230b120e' points='20,240 82,132 144,240'/><polygon fill='%23111c14' points='112,240 192,84 272,240'/><polygon fill='%230b120e' points='220,240 288,120 356,240'/><polygon fill='%2315221a' points='332,240 422,46 512,240'/><polygon fill='%230c130f' points='470,240 538,118 606,240'/><polygon fill='%23111b15' points='560,240 648,74 736,240'/><polygon fill='%230b120e' points='684,240 756,128 828,240'/><polygon fill='%2315221a' points='778,240 868,64 958,240'/></svg>") center bottom / 760px 220px repeat-x;
    opacity: 0.86;
    filter: drop-shadow(0 -10px 22px rgba(0, 0, 0, 0.24));
}

.forest-backdrop .forest-floor {
    position: absolute;
    inset: auto 0 0 0;
    height: 18vh;
    background:
        linear-gradient(180deg, rgba(5, 8, 6, 0), rgba(5, 8, 6, 0.12) 15%, rgba(4, 7, 5, 0.92) 100%),
        repeating-linear-gradient(118deg, rgba(224, 240, 227, 0.025) 0 2px, transparent 2px 44px),
        linear-gradient(180deg, rgba(8, 13, 10, 0.18), rgba(5, 8, 6, 0.95));
}

.story-kicker {
    color: #b8cdbb;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.68rem;
    font-weight: 700;
}

.story-page-divider,
.story-section-divider {
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(239, 245, 240, 0),
        rgba(239, 245, 240, 0.32),
        rgba(239, 245, 240, 0)
    );
}

.story-page-divider {
    margin: 0.85rem 0 1.35rem 0;
}

.story-section-divider {
    margin: 2rem 0 1.25rem 0;
    position: relative;
}

.story-section-divider::after {
    content: "";
    display: block;
    width: 110px;
    height: 16px;
    margin: -8px auto 0 auto;
    background: linear-gradient(135deg, rgba(214, 229, 217, 0.24), rgba(47, 71, 53, 0.12));
    clip-path: polygon(10% 0, 100% 0, 88% 100%, 0 100%);
}

.story-helper-rail,
.story-choice-rail {
    border-left: 1px solid rgba(233, 241, 234, 0.18);
    padding-left: 1rem;
    margin-top: 3.2rem;
}

.story-helper-rail p,
.story-helper-rail li,
.story-choice-rail p,
.story-choice-rail li {
    line-height: 1.55;
}

.story-helper-rail ul,
.story-choice-rail ul {
    margin: 0.65rem 0 0 0;
    padding-left: 1rem;
}

.story-helper-rail code,
.story-choice-rail code {
    color: #f2f7f3;
    background: rgba(233, 241, 234, 0.08);
    border: 1px solid rgba(233, 241, 234, 0.08);
    border-radius: 8px;
    padding: 0.08rem 0.35rem;
}

@media (max-width: 900px) {
    .story-helper-rail,
    .story-choice-rail {
        margin-top: 0.5rem;
    }
}

.st-key-forest_toggle_shell button,
.st-key-story_toggle_shell button {
    background: rgba(9, 14, 11, 0.76);
    border: 1px solid rgba(233, 245, 236, 0.18);
    box-shadow: 0 16px 44px rgba(0, 0, 0, 0.38);
}

.st-key-forest_toggle_shell button p,
.st-key-story_toggle_shell button p {
    color: #edf5ee;
}
</style>
"""

FOREST_THEME_BACKDROP = """
<div class="forest-backdrop" aria-hidden="true">
    <div class="forest-haze"></div>
    <div class="forest-facets-top"></div>
    <div class="forest-ridge-back"></div>
    <div class="forest-ridge-mid"></div>
    <div class="forest-ridge-front"></div>
    <div class="forest-trees-back"></div>
    <div class="forest-trees-front"></div>
    <div class="forest-floor"></div>
</div>
"""


SIDEBAR_RESTORE_HTML = """
<script>
(function () {
  const parentDoc = window.parent.document;
  const sidebarWidth = "21rem";

  function forceSidebarOpen() {
    const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
      sidebar.setAttribute("aria-expanded", "true");
      sidebar.style.minWidth = sidebarWidth;
      sidebar.style.maxWidth = sidebarWidth;
      sidebar.style.width = sidebarWidth;
      sidebar.style.transform = "translateX(0)";
      sidebar.style.marginLeft = "0";
      const sidebarInner = sidebar.querySelector(":scope > div");
      if (sidebarInner) {
        sidebarInner.style.minWidth = sidebarWidth;
        sidebarInner.style.maxWidth = sidebarWidth;
        sidebarInner.style.width = sidebarWidth;
      }
    }

    const openButton = parentDoc.querySelector(
      '[data-testid="stSidebarCollapsedControl"] button, button[aria-label="Open sidebar"]'
    );
    if (openButton) {
      openButton.click();
    }
  }

  forceSidebarOpen();
  window.setTimeout(forceSidebarOpen, 120);
  window.setTimeout(forceSidebarOpen, 500);
})();
</script>
"""


def toggle_forest_theme() -> None:
    st.session_state["forest_theme_enabled"] = not st.session_state.get(
        "forest_theme_enabled",
        False,
    )


def render_theme_mode() -> None:
    forest_theme_enabled = st.session_state.get("forest_theme_enabled", False)
    st.markdown(BASE_THEME_CSS, unsafe_allow_html=True)
    if forest_theme_enabled:
        st.markdown(FOREST_THEME_CSS, unsafe_allow_html=True)
        st.markdown(FOREST_THEME_BACKDROP, unsafe_allow_html=True)


def render_sidebar_restore() -> None:
    components.html(SIDEBAR_RESTORE_HTML, height=0, width=0)


def render_theme_toggle() -> None:
    button_label = "Plain" if st.session_state.get("forest_theme_enabled", False) else "Forest"
    toggle_shell = st.container(key="forest_toggle_shell")
    toggle_shell.button(
        button_label,
        key="forest_theme_button",
        help="Toggle the monochrome low-poly forest scene.",
        on_click=toggle_forest_theme,
    )


def forest_mode_enabled() -> bool:
    return st.session_state.get("forest_theme_enabled", False)


def get_tooltip_selection_tip(key: str) -> str:
    return TOOLTIP_SELECTION_TIPS.get(key, DEFAULT_TOOLTIP_SELECTION_TIP)


def get_story_control_selection_tip(control: str) -> str:
    return STORY_CONTROL_SELECTION_TIPS.get(control, DEFAULT_TOOLTIP_SELECTION_TIP)


def with_story_reference(
    help_text: str,
    story_reference: str | None = None,
    selection_tip: str | None = None,
) -> str:
    if forest_mode_enabled():
        sections = [f"What it does: {help_text}"]
        if story_reference:
            sections.append(f"Forest story: {story_reference}")
        sections.append(
            f"How to choose it: {selection_tip or DEFAULT_TOOLTIP_SELECTION_TIP}"
        )
        return "\n\n".join(sections)
    return help_text


def tooltip_text(key: str) -> str:
    return with_story_reference(
        TOOLTIPS[key],
        STORY_TOOLTIP_LINKS.get(key),
        get_tooltip_selection_tip(key),
    )


def open_story_page() -> None:
    st.session_state["current_page"] = "story"


def open_main_page() -> None:
    st.session_state["current_page"] = "main"


def open_audit_page() -> None:
    st.session_state["current_page"] = "audit"


def open_model_viewer() -> None:
    st.session_state["workflow_mode"] = "inspect"


def open_training_view() -> None:
    st.session_state["workflow_mode"] = "train"


def render_story_navigation() -> None:
    if not forest_mode_enabled():
        return

    on_story_page = st.session_state.get("current_page", "main") == "story"
    nav_shell = st.container(key="story_toggle_shell")
    nav_shell.button(
        "Back" if on_story_page else "Story",
        key="story_page_button",
        help=(
            "Return to the model builder."
            if on_story_page
            else "Open the mushroom story view that explains how the forest thinks."
        ),
        on_click=open_main_page if on_story_page else open_story_page,
    )


def get_story_document_path() -> Path:
    return Path(__file__).with_name(STORY_DOCUMENT_NAME)


def get_favicon_path() -> Path:
    return Path(__file__).with_name("forest_favicon.svg")


@st.cache_data(show_spinner=False)
def load_story_sections() -> list[dict[str, Any]]:
    story_path = get_story_document_path()
    if not story_path.exists():
        return []

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(story_path) as story_zip:
        document_xml = story_zip.read("word/document.xml")

    root = ET.fromstring(document_xml)
    raw_lines: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", namespace):
        pieces: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t":
                pieces.append(node.text or "")
            elif tag in {"br", "cr"}:
                pieces.append("\n")
        paragraph_text = "".join(pieces)
        if paragraph_text.strip():
            raw_lines.extend(line.rstrip() for line in paragraph_text.splitlines())

    cleaned_lines = [line for line in raw_lines if line.strip()]
    sections: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in cleaned_lines:
        if line == "----":
            continue
        if line in STORY_SECTION_HELPERS:
            if current_heading is not None:
                sections.append(
                    {
                        "heading": current_heading,
                        "lines": current_lines.copy(),
                        "controls": STORY_SECTION_HELPERS.get(current_heading, []),
                    }
                )
            current_heading = line
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append(
            {
                "heading": current_heading,
                "lines": current_lines.copy(),
                "controls": STORY_SECTION_HELPERS.get(current_heading, []),
            }
        )

    return sections


def render_story_section(section: dict[str, Any]) -> None:
    section_columns = st.columns([3.0, 1.15, 1.25], gap="large")
    section_markdown = "\n\n".join(section["lines"])
    helper_items = "".join(
        f"<li><code>{escape(control)}</code> {escape(detail)}</li>"
        for control, detail in section["controls"]
    )
    choice_items = "".join(
        "<li><code>"
        + escape(control)
        + "</code> "
        + escape(get_story_control_selection_tip(control))
        + "</li>"
        for control, _detail in section["controls"]
    )

    with section_columns[0]:
        st.markdown('<div class="story-section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### {section['heading']}")
        st.markdown(section_markdown)

    with section_columns[1]:
        st.markdown(
            f"""
            <div class="story-helper-rail">
                <div class="story-kicker">Control Map</div>
                <ul>
                    {helper_items}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with section_columns[2]:
        st.markdown(
            f"""
            <div class="story-choice-rail">
                <div class="story-kicker">Choosing Values</div>
                <ul>
                    {choice_items}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_story_page() -> None:
    story_sections = load_story_sections()
    if not story_sections:
        st.error(f"Could not read `{STORY_DOCUMENT_NAME}` from the project folder.")
        return

    st.title("Forest Story")
    st.caption(
        "Exact wording loaded from `ForestStory.docx`. The right-side rails map each section back to controls in the model builder and suggest how to choose values."
    )
    st.markdown('<div class="story-page-divider"></div>', unsafe_allow_html=True)

    for section in story_sections:
        render_story_section(section)


def load_training_audit_entries() -> tuple[list[dict[str, Any]], int]:
    parsed_entries: list[dict[str, Any]] = []
    skipped_entries = migrate_legacy_audit_log()
    audit_db_path = get_audit_db_path()
    if not audit_db_path.exists():
        return [], skipped_entries

    with sqlite3.connect(audit_db_path, timeout=30) as audit_conn:
        stored_entries = audit_conn.execute(
            """
            SELECT entry_json
            FROM training_audit_entries
            ORDER BY logged_at DESC, id DESC
            """
        ).fetchall()

    for stored_entry_json, in stored_entries:
        try:
            parsed_entry = json.loads(stored_entry_json)
        except json.JSONDecodeError:
            skipped_entries += 1
            continue
        if isinstance(parsed_entry, dict):
            parsed_entries.append(parsed_entry)
        else:
            skipped_entries += 1

    return parsed_entries, skipped_entries


def build_audit_summary_rows(entries: list[dict[str, Any]]) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        dataset_summary = entry.get("dataset_summary", {})
        metrics = entry.get("metrics", {})
        summary_rows.append(
            {
                "Run": index,
                "Logged at": entry.get("logged_at"),
                "Problem type": entry.get("problem_type"),
                "Target": entry.get("target_column"),
                "Features": dataset_summary.get("selected_feature_count"),
                "Train rows": dataset_summary.get("train_rows"),
                "Test rows": dataset_summary.get("test_rows"),
                "MAE": metrics.get("mae"),
                "RMSE": metrics.get("rmse"),
                "R2": metrics.get("r2"),
            }
        )
    return pd.DataFrame(summary_rows)


def format_audit_entry_label(entry: dict[str, Any]) -> str:
    dataset_summary = entry.get("dataset_summary", {})
    return (
        f"{entry.get('logged_at', 'Unknown time')} | "
        f"{entry.get('problem_type', 'unknown')} | "
        f"{entry.get('target_column', 'unknown target')} | "
        f"{dataset_summary.get('selected_feature_count', '?')} features"
    )


def render_audit_log_page() -> None:
    audit_db_path = get_audit_db_path()
    entries, skipped_entries = load_training_audit_entries()

    header_columns = st.columns([4.6, 1.4], gap="small")
    with header_columns[0]:
        st.title("Training Audit Log")
        st.caption(
            "Review saved training runs, inspect all logged variables, and export the audit history."
        )
    with header_columns[1]:
        st.write("")
        st.write("")
        st.button(
            "Back To Inspect"
            if st.session_state.get("workflow_mode") == "inspect"
            else "Back To Training",
            use_container_width=True,
            key="audit_log_back_button",
            on_click=open_main_page,
        )

    if not audit_db_path.exists():
        st.info("No training audit history exists yet. Train a model once to create the local audit database.")
        return

    action_columns = st.columns([1.4, 5], gap="small")
    with action_columns[0]:
        st.download_button(
            "Download Export (.jsonl)",
            data=build_training_audit_export(entries),
            file_name=AUDIT_EXPORT_FILENAME,
            mime="application/json",
            use_container_width=True,
        )
    with action_columns[1]:
        st.caption(f"Local audit store: `{audit_db_path}`")

    if skipped_entries:
        st.warning(
            f"Skipped {skipped_entries} malformed audit entr"
            f"{'y' if skipped_entries == 1 else 'ies'} while loading the audit store."
        )

    if not entries:
        st.info("The audit database exists, but no readable entries were found.")
        return

    st.markdown("**Run summary**")
    st.dataframe(
        build_audit_summary_rows(entries),
        use_container_width=True,
        hide_index=True,
    )

    selected_index = st.selectbox(
        "Audit entry",
        options=list(range(len(entries))),
        format_func=lambda index: format_audit_entry_label(entries[int(index)]),
        key="audit_log_selected_entry",
    )
    selected_entry = entries[int(selected_index)]

    detail_columns = st.columns([1.4, 5], gap="small")
    with detail_columns[0]:
        st.download_button(
            "Download Entry (.json)",
            data=json.dumps(to_json_compatible(selected_entry), indent=2).encode("utf-8"),
            file_name=f"audit_entry_{int(selected_index) + 1}.json",
            mime="application/json",
            use_container_width=True,
        )
    with detail_columns[1]:
        st.caption("Selected entry details")
    st.json(selected_entry, expanded=False)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def load_pickle_artifact(file_bytes: bytes) -> Any:
    return pickle.loads(file_bytes)


def unpack_model_artifact(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        metadata = {
            "problem_type": artifact.get("problem_type"),
            "target_column": artifact.get("target_column"),
            "feature_columns": artifact.get("feature_columns"),
            "input_feature_columns": artifact.get("input_feature_columns"),
            "input_feature_schema": artifact.get("input_feature_schema"),
            "parameters": artifact.get("parameters", {}),
            "target_details": artifact.get("target_details", {"is_datetime_target": False}),
            "audit_log_entry": artifact.get("audit_log_entry"),
        }
    elif isinstance(artifact, Pipeline):
        pipeline = artifact
        metadata = {
            "problem_type": None,
            "target_column": None,
            "feature_columns": None,
            "input_feature_columns": None,
            "input_feature_schema": None,
            "parameters": {},
            "target_details": {"is_datetime_target": False},
            "audit_log_entry": None,
        }
    elif isinstance(artifact, (RandomForestClassifier, RandomForestRegressor)):
        pipeline = None
        metadata = {
            "problem_type": "classification"
            if isinstance(artifact, RandomForestClassifier)
            else "regression",
            "target_column": None,
            "feature_columns": None,
            "input_feature_columns": None,
            "input_feature_schema": None,
            "parameters": artifact.get_params(),
            "target_details": {"is_datetime_target": False},
            "audit_log_entry": None,
        }
        return {
            "pipeline": None,
            "preprocessor": None,
            "model": artifact,
            "metadata": metadata,
        }
    else:
        raise ValueError(
            "Unsupported pickle structure. Upload a `.pkl` file exported by this app, or a saved scikit-learn random forest pipeline."
        )

    if not isinstance(pipeline, Pipeline):
        raise ValueError("The uploaded artifact does not contain a supported scikit-learn pipeline.")
    if "model" not in pipeline.named_steps:
        raise ValueError("The uploaded pipeline does not contain a `model` step.")

    model = pipeline.named_steps["model"]
    if not isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        raise ValueError("The uploaded pipeline model is not a supported random forest.")

    return {
        "pipeline": pipeline,
        "preprocessor": pipeline.named_steps.get("preprocessor"),
        "model": model,
        "metadata": metadata,
    }


def get_loaded_model_feature_names(model_bundle: dict[str, Any]) -> list[str]:
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass

    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    feature_count = int(getattr(model, "n_features_in_", 0))
    return [f"feature_{index}" for index in range(feature_count)]


def get_saved_model_input_feature_names(model_bundle: dict[str, Any]) -> list[str]:
    metadata_input_schema = model_bundle["metadata"].get("input_feature_schema")
    if metadata_input_schema:
        return [str(spec["name"]) for spec in metadata_input_schema]

    metadata_input_features = model_bundle["metadata"].get("input_feature_columns")
    if metadata_input_features:
        return [str(name) for name in metadata_input_features]

    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]

    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return [str(name) for name in preprocessor.feature_names_in_]

    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    metadata_features = model_bundle["metadata"].get("feature_columns")
    if metadata_features:
        return [str(name) for name in metadata_features]

    feature_count = int(getattr(model, "n_features_in_", 0))
    return [f"feature_{index}" for index in range(feature_count)]


def prettify_feature_text(value: str) -> str:
    return value.replace("__", " ").replace("_", " ")


def get_original_feature_names(model_bundle: dict[str, Any]) -> list[str]:
    metadata_features = model_bundle["metadata"].get("feature_columns")
    if metadata_features:
        return [str(name) for name in metadata_features]

    preprocessor = model_bundle["preprocessor"]
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return [str(name) for name in preprocessor.feature_names_in_]

    return []


def get_preprocessor_column_names(
    columns: Any,
    available_columns: list[str],
) -> list[str]:
    if isinstance(columns, slice):
        return available_columns[columns]
    if isinstance(columns, (str, int)):
        return [str(columns)]
    try:
        return [str(column) for column in list(columns)]
    except TypeError:
        return []


def humanize_feature_name(raw_name: str, original_feature_names: list[str]) -> str:
    base_name = raw_name
    for prefix in ("num__", "cat__", "remainder__"):
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break

    for original_name in sorted(original_feature_names, key=len, reverse=True):
        if base_name == original_name:
            return prettify_feature_text(original_name)
        if base_name.startswith(f"{original_name}__"):
            suffix = base_name[len(original_name) + 2 :]
            return f"{prettify_feature_text(original_name)}: {prettify_feature_text(suffix)}"
        if base_name.startswith(f"{original_name}_"):
            category = base_name[len(original_name) + 1 :]
            return f"{prettify_feature_text(original_name)} = {prettify_feature_text(category)}"

    if "__" in base_name:
        left, right = base_name.split("__", 1)
        return f"{prettify_feature_text(left)}: {prettify_feature_text(right)}"

    return prettify_feature_text(base_name)


def get_human_readable_feature_labels(
    model_bundle: dict[str, Any],
    feature_names: list[str],
) -> list[str]:
    original_feature_names = get_original_feature_names(model_bundle)
    return [
        humanize_feature_name(feature_name, original_feature_names)
        for feature_name in feature_names
    ]


def get_saved_model_input_specs(model_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_input_schema = model_bundle["metadata"].get("input_feature_schema")
    if metadata_input_schema:
        return [
            {
                "name": str(spec["name"]),
                "label": str(spec.get("label") or spec["name"]),
                "kind": str(spec.get("kind", "unknown")),
                "options": list(spec.get("options", [])),
            }
            for spec in metadata_input_schema
        ]

    input_feature_names = get_saved_model_input_feature_names(model_bundle)
    original_feature_names = get_original_feature_names(model_bundle)
    preprocessor = model_bundle["preprocessor"]
    default_kind = "unknown" if preprocessor is not None else "numeric"
    kind_lookup = {feature_name: default_kind for feature_name in input_feature_names}
    option_lookup: dict[str, list[Any]] = {feature_name: [] for feature_name in input_feature_names}

    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue

            column_names = get_preprocessor_column_names(columns, input_feature_names)
            normalized_transformer_name = str(transformer_name).lower()
            if normalized_transformer_name.startswith("num"):
                kind = "numeric"
            elif normalized_transformer_name.startswith("cat"):
                kind = "categorical"
            else:
                kind = "unknown"

            for column_name in column_names:
                kind_lookup[column_name] = kind

            encoder = None
            if isinstance(transformer, Pipeline):
                encoder = transformer.named_steps.get("encoder")
            elif hasattr(transformer, "categories_"):
                encoder = transformer

            if kind == "categorical" and encoder is not None and hasattr(encoder, "categories_"):
                for column_name, categories in zip(column_names, encoder.categories_):
                    cleaned_options: list[Any] = []
                    for category in categories:
                        if pd.isna(category):
                            continue
                        value = category.item() if isinstance(category, np.generic) else category
                        if value not in cleaned_options:
                            cleaned_options.append(value)
                    option_lookup[column_name] = cleaned_options

    return [
        {
            "name": feature_name,
            "label": humanize_feature_name(feature_name, original_feature_names),
            "kind": kind_lookup.get(feature_name, default_kind),
            "options": option_lookup.get(feature_name, []),
        }
        for feature_name in input_feature_names
    ]


def build_manual_prediction_input_frame(
    feature_specs: list[dict[str, Any]],
    raw_values: dict[str, Any],
) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for spec in feature_specs:
        raw_value = raw_values.get(spec["name"])
        kind = spec["kind"]

        if kind == "categorical":
            row[spec["name"]] = np.nan if raw_value in (None, "") else raw_value
            continue

        if kind == "numeric":
            text_value = str(raw_value or "").strip()
            if not text_value:
                row[spec["name"]] = np.nan
                continue
            try:
                row[spec["name"]] = float(text_value)
            except ValueError as exc:
                raise ValueError(f"{spec['label']} must be a number.") from exc
            continue

        if kind == "datetime":
            text_value = str(raw_value or "").strip()
            if not text_value:
                row[spec["name"]] = pd.NaT
                continue
            try:
                row[spec["name"]] = pd.to_datetime(text_value, errors="raise")
            except Exception as exc:
                raise ValueError(f"{spec['label']} must be a valid date or date-time.") from exc
            continue

        text_value = str(raw_value or "").strip()
        row[spec["name"]] = np.nan if not text_value else raw_value

    return pd.DataFrame([row], columns=[spec["name"] for spec in feature_specs])


def get_model_estimator_input(model_bundle: dict[str, Any], input_frame: pd.DataFrame) -> Any:
    pipeline = model_bundle["pipeline"]
    if pipeline is not None:
        transformed_input: Any = input_frame
        for step_name, step in list(pipeline.named_steps.items())[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(
                    f"The uploaded pipeline step `{step_name}` does not support transform()."
                )
            transformed_input = step.transform(transformed_input)
        return transformed_input

    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]

    if preprocessor is not None:
        return preprocessor.transform(input_frame)
    if hasattr(model, "feature_names_in_"):
        return input_frame
    return input_frame.to_numpy()


def get_saved_model_prediction_results(
    model_bundle: dict[str, Any],
    input_frame: pd.DataFrame,
) -> dict[str, Any]:
    pipeline = model_bundle["pipeline"]
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    target_details = metadata.get("target_details", {"is_datetime_target": False})

    if pipeline is not None:
        predictor: Any = pipeline
        predictor_input: Any = input_frame
    elif hasattr(model, "feature_names_in_"):
        predictor = model
        predictor_input = input_frame
    else:
        predictor = model
        predictor_input = input_frame.to_numpy()

    predictions = predictor.predict(predictor_input)
    estimator_input = get_model_estimator_input(model_bundle, input_frame)

    if isinstance(model, RandomForestClassifier):
        predicted_label = str(predictions[0])
        probability_table = pd.DataFrame()
        confidence = None
        if hasattr(predictor, "predict_proba"):
            probabilities = predictor.predict_proba(predictor_input)[0]
            probability_table = pd.DataFrame(
                {
                    "class": [str(class_label) for class_label in model.classes_],
                    "probability": probabilities,
                }
            ).sort_values("probability", ascending=False, ignore_index=True)
            if not probability_table.empty:
                confidence = float(probability_table.iloc[0]["probability"])

        tree_votes = pd.Series(
            [str(estimator.predict(estimator_input)[0]) for estimator in model.estimators_],
            dtype="string",
        )
        vote_table = (
            tree_votes.value_counts(dropna=False)
            .rename_axis("prediction")
            .reset_index(name="votes")
            .sort_values(["votes", "prediction"], ascending=[False, True], ignore_index=True)
        )
        vote_table["share"] = vote_table["votes"] / max(len(model.estimators_), 1)

        return {
            "problem_type": "classification",
            "display_prediction": predicted_label,
            "confidence": confidence,
            "winning_votes": int(vote_table.iloc[0]["votes"]) if not vote_table.empty else 0,
            "probability_table": probability_table,
            "vote_table": vote_table,
        }

    predicted_value = float(np.ravel(predictions)[0])
    tree_outputs = np.array(
        [float(np.ravel(estimator.predict(estimator_input))[0]) for estimator in model.estimators_],
        dtype="float64",
    )
    tree_summary = {
        "Mean": predicted_value,
        "Median": float(np.median(tree_outputs)),
        "Min": float(np.min(tree_outputs)),
        "Max": float(np.max(tree_outputs)),
        "Std dev": float(np.std(tree_outputs)),
    }

    if target_details.get("is_datetime_target"):
        summary_rows: list[dict[str, str]] = []
        for label, value in tree_summary.items():
            if label == "Std dev":
                summary_rows.append({"Statistic": label, "Value": f"{value:.3f} days"})
            else:
                summary_rows.append(
                    {
                        "Statistic": label,
                        "Value": format_prediction_for_display(value, metadata),
                    }
                )
        return {
            "problem_type": "regression",
            "display_prediction": format_prediction_for_display(predicted_value, metadata),
            "secondary_display": format_prediction_for_display(tree_summary["Median"], metadata),
            "spread_display": f"{tree_summary['Std dev']:.3f} days",
            "tree_summary": pd.DataFrame(summary_rows),
            "target_is_datetime": True,
        }

    return {
        "problem_type": "regression",
        "display_prediction": f"{predicted_value:.4f}",
        "secondary_display": f"{tree_summary['Median']:.4f}",
        "spread_display": f"{tree_summary['Std dev']:.4f}",
        "tree_summary": pd.DataFrame(
            {
                "Statistic": list(tree_summary.keys()),
                "Value": list(tree_summary.values()),
            }
        ),
        "target_is_datetime": False,
    }


def render_saved_model_prediction_tester(model_bundle: dict[str, Any]) -> None:
    feature_specs = get_saved_model_input_specs(model_bundle)
    if not feature_specs:
        st.info("This saved model does not expose a usable input schema for manual test predictions.")
        return

    st.markdown("**Test the saved forest**")
    st.caption(
        "Enter one example row using the variables this saved model expects. Leave a field blank to let the pipeline impute it when possible."
    )
    if any(spec["kind"] == "unknown" for spec in feature_specs):
        st.info(
            "Some saved inputs do not expose a clear type from the artifact, so those fields use free-text entry."
        )

    raw_values: dict[str, Any] = {}
    with st.form("saved_model_test_prediction_form", clear_on_submit=False):
        input_columns = st.columns(3, gap="medium")
        for index, spec in enumerate(feature_specs):
            with input_columns[index % len(input_columns)]:
                widget_key = f"inspect_manual_input_{spec['name']}"
                if spec["kind"] == "categorical" and spec["options"]:
                    raw_values[spec["name"]] = st.selectbox(
                        spec["label"],
                        options=[None, *spec["options"]],
                        index=0,
                        format_func=(
                            lambda value: "Leave blank / missing"
                            if value is None
                            else str(value)
                        ),
                        key=widget_key,
                    )
                else:
                    raw_values[spec["name"]] = st.text_input(
                        spec["label"],
                        key=widget_key,
                        placeholder="Leave blank / missing",
                    )

        submitted = st.form_submit_button("Run Test Prediction", type="primary")

    if not submitted:
        return

    try:
        input_frame = build_manual_prediction_input_frame(feature_specs, raw_values)
        prediction_results = get_saved_model_prediction_results(model_bundle, input_frame)
    except Exception as exc:
        st.error(f"Could not run the test prediction: {exc}")
        return

    label_lookup = {spec["name"]: spec["label"] for spec in feature_specs}

    if prediction_results["problem_type"] == "classification":
        result_columns = st.columns(3)
        result_columns[0].metric("Predicted class", prediction_results["display_prediction"])
        confidence = prediction_results.get("confidence")
        result_columns[1].metric(
            "Confidence",
            f"{confidence * 100:.1f}%" if confidence is not None else "Unavailable",
        )
        result_columns[2].metric(
            "Trees backing winner",
            f"{prediction_results['winning_votes']} / {len(model_bundle['model'].estimators_)}",
        )

        detail_columns = st.columns(2)
        with detail_columns[0]:
            st.markdown("**Class probabilities**")
            if prediction_results["probability_table"].empty:
                st.info("This saved model does not expose class probabilities.")
            else:
                st.dataframe(
                    prediction_results["probability_table"],
                    use_container_width=True,
                    hide_index=True,
                )
        with detail_columns[1]:
            st.markdown("**Tree vote breakdown**")
            st.dataframe(
                prediction_results["vote_table"],
                use_container_width=True,
                hide_index=True,
            )
    else:
        target_is_datetime = prediction_results.get("target_is_datetime", False)
        result_columns = st.columns(3)
        result_columns[0].metric(
            "Predicted date" if target_is_datetime else "Predicted value",
            prediction_results["display_prediction"],
        )
        result_columns[1].metric(
            "Median tree output",
            prediction_results["secondary_display"],
        )
        result_columns[2].metric(
            "Tree spread",
            prediction_results["spread_display"],
        )
        st.markdown("**Tree output summary**")
        st.dataframe(
            prediction_results["tree_summary"],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Prediction input row", expanded=False):
        st.dataframe(
            input_frame.rename(columns=label_lookup),
            use_container_width=True,
            hide_index=True,
        )


def get_split_branch_labels(feature_label: str, threshold: float) -> tuple[str, str, str]:
    if " = " in feature_label and abs(threshold - 0.5) <= 0.05:
        field_name, field_value = feature_label.split(" = ", 1)
        question = f"Is {field_name} {field_value}?"
        return question, f"{field_name} is not {field_value}", f"{field_name} is {field_value}"

    question = f"Is {feature_label} <= {threshold:.2f}?"
    return question, f"{feature_label} <= {threshold:.2f}", f"{feature_label} > {threshold:.2f}"


def format_prediction_for_display(value: float, metadata: dict[str, Any]) -> str:
    target_details = metadata.get("target_details", {"is_datetime_target": False})
    if target_details.get("is_datetime_target"):
        decoded = decode_datetime_predictions(
            np.array([value]),
            includes_time=target_details.get("includes_time", False),
        )
        return format_datetime_output(
            decoded,
            target_details.get("display_format", "%Y-%m-%d"),
        ).iloc[0]

    return f"{value:.3f}"


def get_leaf_prediction_summary(
    model: RandomForestClassifier | RandomForestRegressor,
    metadata: dict[str, Any],
    estimator: Any,
    node_index: int,
) -> dict[str, str]:
    tree = estimator.tree_
    sample_count = int(round(float(tree.weighted_n_node_samples[node_index])))

    if isinstance(model, RandomForestClassifier):
        class_counts = np.ravel(tree.value[node_index]).astype(float)
        total_count = float(class_counts.sum())
        if total_count <= 0:
            class_probabilities = np.zeros_like(class_counts)
        else:
            class_probabilities = class_counts / total_count

        predicted_index = int(np.argmax(class_counts))
        predicted_label = str(model.classes_[predicted_index])
        confidence = float(class_probabilities[predicted_index])
        ranked_classes = np.argsort(class_probabilities)[::-1]
        class_mix = ", ".join(
            f"{model.classes_[class_index]} {class_probabilities[class_index] * 100:.0f}%"
            for class_index in ranked_classes[: min(3, len(ranked_classes))]
        )
        return {
            "graph": (
                f"Outcome: {predicted_label}\n"
                f"Confidence: {confidence * 100:.0f}%\n"
                f"Samples: {sample_count}"
            ),
            "text": (
                f"Predict {predicted_label} with about {confidence * 100:.0f}% confidence "
                f"from {sample_count} sample(s). Class mix: {class_mix}."
            ),
        }

    predicted_value = float(np.ravel(tree.value[node_index])[0])
    formatted_value = format_prediction_for_display(predicted_value, metadata)
    return {
        "graph": f"Outcome: {formatted_value}\nSamples: {sample_count}",
        "text": f"Predict {formatted_value} based on {sample_count} sample(s).",
    }


def get_tree_importance_score(estimator: Any) -> float:
    tree = estimator.tree_
    total_gain = 0.0
    for node_index in range(tree.node_count):
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]
        if left_child == right_child:
            continue
        parent_weight = float(tree.weighted_n_node_samples[node_index])
        left_weight = float(tree.weighted_n_node_samples[left_child])
        right_weight = float(tree.weighted_n_node_samples[right_child])
        gain = (
            parent_weight * float(tree.impurity[node_index])
            - left_weight * float(tree.impurity[left_child])
            - right_weight * float(tree.impurity[right_child])
        )
        if gain > 0:
            total_gain += gain
    return total_gain


def get_forest_statistics(model: RandomForestClassifier | RandomForestRegressor) -> pd.DataFrame:
    rows: list[dict[str, int | float]] = []
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        rows.append(
            {
                "tree_index": index,
                "depth": int(tree.max_depth),
                "leaves": int(tree.n_leaves),
                "nodes": int(tree.node_count),
                "importance_score": get_tree_importance_score(estimator),
            }
        )

    forest_stats = pd.DataFrame(rows)
    total_importance = float(forest_stats["importance_score"].sum())
    if total_importance > 0:
        forest_stats["weight"] = forest_stats["importance_score"] / total_importance
    else:
        forest_stats["weight"] = 1.0 / max(len(forest_stats), 1)

    forest_stats["importance_rank"] = (
        forest_stats["importance_score"].rank(method="dense", ascending=False).astype(int)
    )
    return forest_stats.sort_values(["importance_rank", "tree_index"]).reset_index(drop=True)


def get_model_overview_rows(
    model_bundle: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    metadata = model_bundle["metadata"]
    model = model_bundle["model"]
    overview = [
        {"Property": "Model type", "Value": metadata.get("problem_type") or type(model).__name__},
        {"Property": "Trees", "Value": str(len(model.estimators_))},
        {"Property": "Target column", "Value": str(metadata.get("target_column") or "Not stored")},
        {"Property": "Feature count", "Value": str(len(feature_names))},
        {"Property": "Bootstrap", "Value": str(getattr(model, "bootstrap", "Unknown"))},
        {"Property": "Criterion", "Value": str(getattr(model, "criterion", "Unknown"))},
    ]
    return pd.DataFrame(overview)


def to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return [
            to_json_compatible(record)
            for record in value.to_dict(orient="records")
        ]
    if isinstance(value, pd.Series):
        return [to_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [to_json_compatible(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def build_model_metadata_export(
    model_bundle: dict[str, Any],
    feature_names: list[str],
    feature_labels: list[str],
    importance_table: pd.DataFrame,
    forest_stats: pd.DataFrame,
) -> bytes:
    model = model_bundle["model"]
    export_payload = {
        "overview": get_model_overview_rows(model_bundle, feature_names),
        "metadata": model_bundle["metadata"],
        "resolved_feature_names": feature_names,
        "resolved_feature_labels": feature_labels,
        "forest_summary": {
            "tree_count": len(model.estimators_),
            "average_depth": float(forest_stats["depth"].mean()),
            "average_leaves": float(forest_stats["leaves"].mean()),
            "feature_count": len(feature_names),
        },
        "feature_importances": importance_table,
        "per_tree_statistics": forest_stats.sort_values("tree_index"),
    }
    serialized_payload = to_json_compatible(export_payload)
    return json.dumps(serialized_payload, indent=2).encode("utf-8")


def get_audit_db_path() -> Path:
    return Path(__file__).with_name(AUDIT_DB_FILENAME)


def get_legacy_audit_log_path() -> Path:
    return Path(__file__).with_name(LEGACY_AUDIT_LOG_FILENAME)


def initialize_audit_storage() -> Path:
    audit_db_path = get_audit_db_path()
    with sqlite3.connect(audit_db_path, timeout=30) as audit_conn:
        audit_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                entry_hash TEXT NOT NULL UNIQUE,
                entry_json TEXT NOT NULL
            )
            """
        )
        audit_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_audit_entries_logged_at
            ON training_audit_entries (logged_at DESC, id DESC)
            """
        )
        audit_conn.commit()
    return audit_db_path


def serialize_training_audit_entry(entry: dict[str, Any]) -> str:
    return json.dumps(to_json_compatible(entry), ensure_ascii=True, sort_keys=True)


def insert_training_audit_entry(
    audit_conn: sqlite3.Connection, entry: dict[str, Any]
) -> bool:
    serialized_entry = serialize_training_audit_entry(entry)
    entry_hash = hashlib.sha256(serialized_entry.encode("utf-8")).hexdigest()
    logged_at = str(entry.get("logged_at", ""))
    cursor = audit_conn.execute(
        """
        INSERT OR IGNORE INTO training_audit_entries (logged_at, entry_hash, entry_json)
        VALUES (?, ?, ?)
        """,
        (logged_at, entry_hash, serialized_entry),
    )
    return cursor.rowcount > 0


def migrate_legacy_audit_log() -> int:
    legacy_audit_log_path = get_legacy_audit_log_path()
    if not legacy_audit_log_path.exists():
        return 0

    audit_db_path = initialize_audit_storage()
    skipped_entries = 0
    with sqlite3.connect(audit_db_path, timeout=30) as audit_conn:
        with legacy_audit_log_path.open("r", encoding="utf-8") as audit_handle:
            for line in audit_handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    parsed_entry = json.loads(stripped_line)
                except json.JSONDecodeError:
                    skipped_entries += 1
                    continue
                if isinstance(parsed_entry, dict):
                    insert_training_audit_entry(audit_conn, parsed_entry)
                else:
                    skipped_entries += 1
        audit_conn.commit()
    return skipped_entries


def build_training_audit_export(entries: list[dict[str, Any]]) -> bytes:
    serialized_entries = [
        serialize_training_audit_entry(entry) for entry in entries if isinstance(entry, dict)
    ]
    if not serialized_entries:
        return b""
    return ("\n".join(serialized_entries) + "\n").encode("utf-8")


def get_metric_value_lookup(metric_table: pd.DataFrame) -> dict[str, Any]:
    metric_lookup: dict[str, Any] = {}
    for record in metric_table.to_dict(orient="records"):
        metric_name = str(record.get("Metric", ""))
        metric_value = record.get("Value")
        if isinstance(metric_value, np.generic):
            metric_value = metric_value.item()
        metric_lookup[metric_name] = metric_value
    return metric_lookup


def build_training_audit_entry(
    *,
    problem_type: str,
    target_column: str,
    feature_columns: list[str],
    input_feature_columns: list[str],
    input_feature_schema: list[dict[str, Any]],
    date_columns: list[str],
    date_dayfirst: bool,
    custom_column_specs: list[dict[str, Any]],
    required_custom_specs: list[dict[str, Any]],
    parameter_summary: dict[str, Any],
    target_details: dict[str, Any],
    metric_table: pd.DataFrame,
    model_input_row_count: int,
    missing_target_rows: int,
    train_row_count: int,
    test_row_count: int,
    test_size: float,
    random_state: int,
    shuffle_rows: bool,
    stratify_split: bool,
    cv_summary: dict[str, Any],
) -> dict[str, Any]:
    metric_lookup = get_metric_value_lookup(metric_table)
    return {
        "logged_at": datetime.now().astimezone().isoformat(),
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_columns": [str(name) for name in feature_columns],
        "input_feature_columns": [str(name) for name in input_feature_columns],
        "input_feature_schema": input_feature_schema,
        "date_columns": [str(name) for name in date_columns],
        "date_dayfirst": bool(date_dayfirst),
        "custom_columns": {
            "defined": custom_column_specs,
            "applied_to_model": required_custom_specs,
        },
        "dataset_summary": {
            "rows_after_target_drop": int(model_input_row_count),
            "missing_target_rows_dropped": int(missing_target_rows),
            "selected_feature_count": len(feature_columns),
            "input_feature_count": len(input_feature_columns),
            "train_rows": int(train_row_count),
            "test_rows": int(test_row_count),
        },
        "train_test_split": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "shuffle_rows": bool(shuffle_rows),
            "stratify_split": bool(stratify_split),
        },
        "parameters": parameter_summary,
        "target_details": target_details,
        "metrics": {
            "displayed_metrics": metric_lookup,
            "mae": metric_lookup.get("MAE", metric_lookup.get("MAE (days)")),
            "rmse": metric_lookup.get("RMSE", metric_lookup.get("RMSE (days)")),
            "r2": metric_lookup.get("R2"),
            "unit": "days" if target_details.get("is_datetime_target") else None,
        },
        "cross_validation": cv_summary,
    }


def append_training_audit_log(entry: dict[str, Any]) -> Path:
    audit_db_path = initialize_audit_storage()
    with sqlite3.connect(audit_db_path, timeout=30) as audit_conn:
        insert_training_audit_entry(audit_conn, entry)
        audit_conn.commit()
    return audit_db_path


def get_importance_table_from_feature_names(
    model: RandomForestClassifier | RandomForestRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    if len(feature_names) != len(model.feature_importances_):
        feature_names = [f"feature_{index}" for index in range(len(model.feature_importances_))]
    importance_table = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    return importance_table.sort_values("importance", ascending=False)


def wrap_display_text(value: str, *, width: int = 22) -> list[str]:
    wrapped_lines: list[str] = []
    for source_line in str(value).splitlines():
        segments = textwrap.wrap(
            source_line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(segments or [""])
    return wrapped_lines or [""]


def build_tree_svg_html(
    model: RandomForestClassifier | RandomForestRegressor,
    *,
    metadata: dict[str, Any],
    feature_names: list[str],
    feature_labels: list[str],
    tree_index: int,
    max_depth: int,
    dark_mode: bool,
) -> str:
    del feature_names
    estimator = model.estimators_[tree_index]
    tree = estimator.tree_
    visible_max_depth = min(max_depth, int(tree.max_depth))
    slot_width = 194.0
    level_height = 176.0
    node_width = 176.0
    label_width = 134.0
    line_height = 15.0
    node_height_padding = 34.0
    label_line_height = 13.0
    label_height_padding = 18.0
    margin_x = 56.0
    margin_y = 52.0

    palette = {
        "frame": "#09130e" if dark_mode else "#eef4ef",
        "panel": "#0f1b14" if dark_mode else "#f8fbf8",
        "panel_border": "#365243" if dark_mode else "#cad8cd",
        "text": "#eef6ef" if dark_mode else "#172019",
        "muted": "#bed0c2" if dark_mode else "#55625a",
        "split_fill": "#173022" if dark_mode else "#ffffff",
        "split_stroke": "#5f8a6d" if dark_mode else "#a9b9ad",
        "leaf_fill": "#294333" if dark_mode else "#e4eee6",
        "leaf_stroke": "#7ea38a" if dark_mode else "#9fb4a5",
        "edge": "#83a08c" if dark_mode else "#9aaca0",
        "edge_label_fill": "#122019" if dark_mode else "#ffffff",
        "edge_label_stroke": "#3d5b48" if dark_mode else "#ccd8cf",
        "edge_label_text": "#eff6f0" if dark_mode else "#223027",
        "shadow": "rgba(0, 0, 0, 0.28)" if dark_mode else "rgba(16, 24, 18, 0.12)",
    }

    visible_leaf_cache: dict[int, int] = {}

    def count_visible_leaves(node_index: int, depth: int) -> int:
        if node_index in visible_leaf_cache:
            return visible_leaf_cache[node_index]
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]
        if left_child == right_child or depth >= max_depth:
            visible_leaf_cache[node_index] = 1
        else:
            visible_leaf_cache[node_index] = count_visible_leaves(
                left_child, depth + 1
            ) + count_visible_leaves(right_child, depth + 1)
        return visible_leaf_cache[node_index]

    node_lookup: dict[int, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    max_seen_depth = 0

    def add_node(node_index: int, depth: int, start_slot: float) -> None:
        nonlocal max_seen_depth
        max_seen_depth = max(max_seen_depth, depth)
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]
        is_leaf = left_child == right_child
        visible_leaf_count = count_visible_leaves(node_index, depth)
        sample_count = int(round(float(tree.weighted_n_node_samples[node_index])))
        center_x = margin_x + (start_slot + visible_leaf_count / 2.0) * slot_width
        y = margin_y + depth * level_height

        if is_leaf or depth >= max_depth:
            leaf_summary = get_leaf_prediction_summary(
                model,
                metadata,
                estimator,
                node_index,
            )
            label = leaf_summary["graph"]
            if not is_leaf and depth >= max_depth:
                label = f"More splits below\n{label}"
            text_lines = wrap_display_text(label, width=19)
            node_kind = "leaf"
        else:
            feature_index = int(tree.feature[node_index])
            question, left_text, right_text = get_split_branch_labels(
                feature_labels[feature_index],
                float(tree.threshold[node_index]),
            )
            text_lines = wrap_display_text(question, width=19)
            text_lines.append(f"Samples: {sample_count}")
            node_kind = "split"

        height = node_height_padding + len(text_lines) * line_height
        node_lookup[node_index] = {
            "node_index": node_index,
            "kind": node_kind,
            "x": center_x - node_width / 2.0,
            "y": y,
            "width": node_width,
            "height": height,
            "center_x": center_x,
            "center_y": y + height / 2.0,
            "text_lines": text_lines,
        }

        if node_kind == "leaf":
            return

        left_feature_span = count_visible_leaves(left_child, depth + 1)
        add_node(left_child, depth + 1, start_slot)
        add_node(right_child, depth + 1, start_slot + left_feature_span)
        edges.append(
            {"parent": node_index, "child": left_child, "label": left_text}
        )
        edges.append(
            {"parent": node_index, "child": right_child, "label": right_text}
        )

    add_node(0, 0, 0.0)
    total_leaf_slots = max(count_visible_leaves(0, 0), 1)
    svg_width = margin_x * 2.0 + total_leaf_slots * slot_width
    svg_height = margin_y * 2.0 + (max_seen_depth + 1) * level_height + 72.0

    edge_markup: list[str] = []
    edge_label_markup: list[str] = []
    for edge in edges:
        parent_node = node_lookup[edge["parent"]]
        child_node = node_lookup[edge["child"]]
        start_x = parent_node["center_x"]
        start_y = parent_node["y"] + parent_node["height"]
        end_x = child_node["center_x"]
        end_y = child_node["y"]
        control_offset = min(46.0, max((end_y - start_y) / 2.0, 24.0))
        path = (
            f"M {start_x:.1f} {start_y:.1f} "
            f"C {start_x:.1f} {start_y + control_offset:.1f}, "
            f"{end_x:.1f} {end_y - control_offset:.1f}, "
            f"{end_x:.1f} {end_y:.1f}"
        )
        edge_markup.append(
            f'<path d="{path}" fill="none" stroke="{palette["edge"]}" '
            'stroke-width="2.2" stroke-linecap="round" />'
        )

        label_lines = wrap_display_text(edge["label"], width=16)
        label_height = label_height_padding + len(label_lines) * label_line_height
        label_x = ((start_x + end_x) / 2.0) - label_width / 2.0
        label_y = ((start_y + end_y) / 2.0) - label_height / 2.0
        tspans = "".join(
            (
                f'<tspan x="{label_x + label_width / 2.0:.1f}" '
                f'y="{label_y + 16.0 + index * label_line_height:.1f}">'
                f"{escape(line)}</tspan>"
            )
            for index, line in enumerate(label_lines)
        )
        edge_label_markup.append(
            f'<g><rect x="{label_x:.1f}" y="{label_y:.1f}" width="{label_width:.1f}" '
            f'height="{label_height:.1f}" rx="16" fill="{palette["edge_label_fill"]}" '
            f'stroke="{palette["edge_label_stroke"]}" stroke-width="1.2" />'
            f'<text text-anchor="middle" font-size="11" font-family="Trebuchet MS, sans-serif" '
            f'font-weight="600" fill="{palette["edge_label_text"]}">{tspans}</text></g>'
        )

    node_markup: list[str] = []
    for node in node_lookup.values():
        fill = palette["split_fill"] if node["kind"] == "split" else palette["leaf_fill"]
        stroke = (
            palette["split_stroke"] if node["kind"] == "split" else palette["leaf_stroke"]
        )
        tspans = "".join(
            (
                f'<tspan x="{node["center_x"]:.1f}" '
                f'y="{node["y"] + 24.0 + index * line_height:.1f}">{escape(line)}</tspan>'
            )
            for index, line in enumerate(node["text_lines"])
        )
        node_markup.append(
            f'<g><rect x="{node["x"]:.1f}" y="{node["y"]:.1f}" width="{node["width"]:.1f}" '
            f'height="{node["height"]:.1f}" rx="18" fill="{fill}" stroke="{stroke}" '
            'stroke-width="1.6" />'
            f'<text text-anchor="middle" font-size="12" font-family="Trebuchet MS, sans-serif" '
            f'font-weight="600" fill="{palette["text"]}">{tspans}</text></g>'
        )

    depth_note = (
        f"Showing the full tree through depth {visible_max_depth}."
        if visible_max_depth >= int(tree.max_depth)
        else f"Showing through depth {visible_max_depth} of {int(tree.max_depth)}."
    )
    stage_height = min(max(visible_max_depth, 4) * 62 + 430, 980)

    return f"""
<div class="rf-tree-viewer">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: {palette["frame"]};
      color: {palette["text"]};
      font-family: "Trebuchet MS", sans-serif;
    }}
    .rf-tree-shell {{
      border: 1px solid {palette["panel_border"]};
      border-radius: 20px;
      overflow: hidden;
      background: linear-gradient(180deg, {palette["panel"]} 0%, {palette["frame"]} 100%);
      box-shadow: 0 18px 40px {palette["shadow"]};
    }}
    .rf-tree-toolbar {{
      display: flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.8rem 0.95rem;
      border-bottom: 1px solid {palette["panel_border"]};
      background: rgba(255, 255, 255, 0.02);
      flex-wrap: wrap;
    }}
    .rf-tree-toolbar button {{
      border: 1px solid {palette["panel_border"]};
      background: {palette["split_fill"]};
      color: {palette["text"]};
      border-radius: 999px;
      padding: 0.42rem 0.72rem;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      cursor: pointer;
    }}
    .rf-tree-toolbar span {{
      color: {palette["muted"]};
      font-size: 0.78rem;
      line-height: 1.45;
    }}
    .rf-tree-stage {{
      height: {stage_height}px;
      overflow: hidden;
    }}
    .rf-tree-stage svg {{
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: grab;
      user-select: none;
    }}
    .rf-tree-stage svg.is-dragging {{
      cursor: grabbing;
    }}
  </style>
  <div class="rf-tree-shell">
    <div class="rf-tree-toolbar">
      <button type="button" id="rf-tree-zoom-in">Zoom In</button>
      <button type="button" id="rf-tree-zoom-out">Zoom Out</button>
      <button type="button" id="rf-tree-reset">Reset</button>
      <span>{escape(depth_note)} Wheel to zoom. Drag to pan.</span>
    </div>
    <div class="rf-tree-stage">
      <svg id="rf-tree-svg" viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" xmlns="http://www.w3.org/2000/svg" aria-label="Random forest decision tree">
        <rect x="0" y="0" width="{svg_width:.1f}" height="{svg_height:.1f}" fill="{palette["frame"]}" />
        <g id="rf-tree-viewport">
          {''.join(edge_markup)}
          {''.join(edge_label_markup)}
          {''.join(node_markup)}
        </g>
      </svg>
    </div>
  </div>
  <script>
    (function () {{
      const svg = document.getElementById("rf-tree-svg");
      const viewport = document.getElementById("rf-tree-viewport");
      const zoomInButton = document.getElementById("rf-tree-zoom-in");
      const zoomOutButton = document.getElementById("rf-tree-zoom-out");
      const resetButton = document.getElementById("rf-tree-reset");
      const viewBox = svg.viewBox.baseVal;
      const minScale = 0.75;
      const maxScale = 12;
      let scale = 1;
      let translateX = 0;
      let translateY = 0;
      let dragging = false;
      let dragPointerId = null;
      let dragX = 0;
      let dragY = 0;

      function applyTransform() {{
        viewport.setAttribute("transform", `translate(${{translateX}} ${{translateY}}) scale(${{scale}})`);
      }}

      function pointToSvg(clientX, clientY) {{
        const point = svg.createSVGPoint();
        point.x = clientX;
        point.y = clientY;
        const matrix = svg.getScreenCTM();
        if (!matrix) {{
          return {{ x: viewBox.width / 2, y: viewBox.height / 2 }};
        }}
        return point.matrixTransform(matrix.inverse());
      }}

      function zoomAt(nextScale, clientX, clientY) {{
        const clampedScale = Math.max(minScale, Math.min(maxScale, nextScale));
        const anchor = pointToSvg(clientX, clientY);
        const worldX = (anchor.x - translateX) / scale;
        const worldY = (anchor.y - translateY) / scale;
        scale = clampedScale;
        translateX = anchor.x - worldX * scale;
        translateY = anchor.y - worldY * scale;
        applyTransform();
      }}

      function zoomBy(factor) {{
        const rect = svg.getBoundingClientRect();
        zoomAt(scale * factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
      }}

      zoomInButton.addEventListener("click", function () {{
        zoomBy(1.2);
      }});

      zoomOutButton.addEventListener("click", function () {{
        zoomBy(1 / 1.2);
      }});

      resetButton.addEventListener("click", function () {{
        scale = 1;
        translateX = 0;
        translateY = 0;
        applyTransform();
      }});

      svg.addEventListener("wheel", function (event) {{
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 / 1.12 : 1.12;
        zoomAt(scale * direction, event.clientX, event.clientY);
      }}, {{ passive: false }});

      svg.addEventListener("pointerdown", function (event) {{
        dragging = true;
        dragPointerId = event.pointerId;
        dragX = event.clientX;
        dragY = event.clientY;
        svg.classList.add("is-dragging");
        svg.setPointerCapture(event.pointerId);
      }});

      svg.addEventListener("pointermove", function (event) {{
        if (!dragging || dragPointerId !== event.pointerId) {{
          return;
        }}
        const deltaX = event.clientX - dragX;
        const deltaY = event.clientY - dragY;
        dragX = event.clientX;
        dragY = event.clientY;
        translateX += (deltaX * viewBox.width) / svg.clientWidth;
        translateY += (deltaY * viewBox.height) / svg.clientHeight;
        applyTransform();
      }});

      function endDrag(event) {{
        if (dragPointerId !== null && event.pointerId !== dragPointerId) {{
          return;
        }}
        dragging = false;
        dragPointerId = null;
        svg.classList.remove("is-dragging");
      }}

      svg.addEventListener("pointerup", endDrag);
      svg.addEventListener("pointercancel", endDrag);
      svg.addEventListener("pointerleave", function (event) {{
        if (dragging && dragPointerId === event.pointerId) {{
          endDrag(event);
        }}
      }});

      applyTransform();
    }})();
  </script>
</div>
"""


def build_tree_rules_html(
    model: RandomForestClassifier | RandomForestRegressor,
    *,
    metadata: dict[str, Any],
    feature_names: list[str],
    feature_labels: list[str],
    tree_index: int,
    max_depth: int,
) -> str:
    del feature_names
    estimator = model.estimators_[tree_index]
    tree = estimator.tree_
    actual_depth = int(tree.max_depth)

    def render_branch(node_index: int, depth: int, summary_text: str) -> str:
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]
        is_leaf = left_child == right_child
        sample_count = int(round(float(tree.weighted_n_node_samples[node_index])))

        if is_leaf or depth >= max_depth:
            leaf_text = get_leaf_prediction_summary(
                model,
                metadata,
                estimator,
                node_index,
            )["text"]
            if not is_leaf and depth >= max_depth:
                leaf_text = f"{leaf_text} More detailed splits continue below this point."
            return (
                "<details>"
                f"<summary>{escape(summary_text)} [{sample_count} samples]</summary>"
                '<div class="tree-rules-body">'
                f'<div class="tree-leaf">{escape(leaf_text)}</div>'
                "</div>"
                "</details>"
            )

        feature_index = int(tree.feature[node_index])
        question, left_text, right_text = get_split_branch_labels(
            feature_labels[feature_index],
            float(tree.threshold[node_index]),
        )
        return (
            "<details>"
            f"<summary>{escape(summary_text)} [{sample_count} samples]</summary>"
            '<div class="tree-rules-body">'
            f'<div class="tree-rules-question">Next question: {escape(question)}</div>'
            f'{render_branch(left_child, depth + 1, f"If {left_text}")}'
            f'{render_branch(right_child, depth + 1, f"Otherwise, if {right_text}")}'
            "</div>"
            "</details>"
        )

    if tree.children_left[0] == tree.children_right[0] or max_depth <= 0:
        root_text = get_leaf_prediction_summary(
            model,
            metadata,
            estimator,
            0,
        )["text"]
        if tree.children_left[0] != tree.children_right[0] and max_depth <= 0:
            root_text = f"{root_text} More detailed splits continue below this point."
        return (
            '<div class="tree-rules">'
            f'<p class="tree-rules-note">Showing only the root of tree {tree_index}. Expand the drawing depth to reveal additional branches.</p>'
            f'<div class="tree-leaf">{escape(root_text)}</div>'
            "</div>"
        )

    root_feature_index = int(tree.feature[0])
    root_question, left_text, right_text = get_split_branch_labels(
        feature_labels[root_feature_index],
        float(tree.threshold[0]),
    )
    scope_note = (
        f"Showing the full tree at depth {actual_depth}."
        if max_depth >= actual_depth
        else f"Showing branches through depth {max_depth} of {actual_depth}."
    )
    root_samples = int(round(float(tree.weighted_n_node_samples[0])))
    return (
        '<div class="tree-rules">'
        f'<p class="tree-rules-note">Tree {tree_index} reader. {escape(scope_note)} Each branch starts collapsed so you can expand only the paths you care about.</p>'
        f'<div class="tree-rules-question">Starting question: {escape(root_question)} [{root_samples} samples]</div>'
        f'{render_branch(tree.children_left[0], 1, f"If {left_text}")}'
        f'{render_branch(tree.children_right[0], 1, f"Otherwise, if {right_text}")}'
        "</div>"
    )


def render_saved_model_viewer() -> None:
    st.subheader("Saved Model Viewer")
    st.caption(
        "Upload a `.pkl` exported by this app to inspect the forest without retraining it."
    )
    uploaded_model = st.file_uploader(
        "Upload a saved model (.pkl)",
        type=["pkl"],
        help=tooltip_text("upload_model"),
        key="saved_model_upload",
    )
    st.warning(
        "Pickle files can run code when loaded. Only open `.pkl` files you created yourself or fully trust."
    )
    trust_model_file = st.checkbox(
        "I trust this `.pkl` file",
        value=False,
        help=tooltip_text("trust_model_file"),
        key="trust_saved_model_file",
    )

    if uploaded_model is None:
        st.info("Upload a `.pkl` file to inspect a saved forest.")
        return
    if not trust_model_file:
        st.info("Confirm that you trust the file before the app loads it.")
        return

    try:
        artifact = load_pickle_artifact(uploaded_model.getvalue())
        model_bundle = unpack_model_artifact(artifact)
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        return

    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    if not hasattr(model, "estimators_") or len(model.estimators_) == 0:
        st.error("The uploaded random forest does not appear to be fitted yet.")
        return
    feature_names = get_loaded_model_feature_names(model_bundle)
    feature_labels = get_human_readable_feature_labels(model_bundle, feature_names)
    importance_table = get_importance_table_from_feature_names(model, feature_names)
    feature_label_lookup = dict(zip(feature_names, feature_labels))
    importance_table = importance_table.copy()
    importance_table["feature"] = importance_table["feature"].map(
        lambda value: feature_label_lookup.get(value, value)
    )
    forest_stats = get_forest_statistics(model)

    st.caption("This viewer works best with `.pkl` files exported by this app.")

    overview_columns = st.columns(4)
    overview_columns[0].metric("Trees", f"{len(model.estimators_):,}")
    overview_columns[1].metric("Avg depth", f"{forest_stats['depth'].mean():.1f}")
    overview_columns[2].metric("Avg leaves", f"{forest_stats['leaves'].mean():.1f}")
    overview_columns[3].metric("Features", f"{len(feature_names):,}")

    with st.expander("Model summary", expanded=True):
        st.dataframe(
            get_model_overview_rows(model_bundle, feature_names),
            use_container_width=True,
            hide_index=True,
        )
        if metadata.get("feature_columns"):
            st.write(
                "Original feature columns: "
                + ", ".join(str(name) for name in metadata["feature_columns"])
            )
        if metadata.get("parameters"):
            st.json(metadata["parameters"], expanded=False)
        if metadata.get("audit_log_entry"):
            st.markdown("**Training audit**")
            st.json(metadata["audit_log_entry"], expanded=False)
        metadata_export_name = f"{Path(uploaded_model.name).stem or 'saved_model'}_metadata.json"
        st.download_button(
            "Export metadata (.json)",
            data=build_model_metadata_export(
                model_bundle,
                feature_names,
                feature_labels,
                importance_table,
                forest_stats,
            ),
            file_name=metadata_export_name,
            mime="application/json",
        )

    stats_columns = st.columns(2)
    with stats_columns[0]:
        st.markdown("**Per-tree statistics**")
        st.dataframe(
            forest_stats,
            use_container_width=True,
            hide_index=True,
            column_config=get_tree_statistics_column_config(),
        )
    with stats_columns[1]:
        st.markdown("**Tree depth and leaves**")
        depth_leaf_chart = forest_stats.sort_values("tree_index").set_index("tree_index")[
            ["depth", "leaves"]
        ]
        st.line_chart(depth_leaf_chart)

    importance_columns = st.columns(2)
    with importance_columns[0]:
        st.markdown("**Feature importances**")
        st.dataframe(
            importance_table.head(30),
            use_container_width=True,
            hide_index=True,
            column_config=get_feature_importance_column_config(),
        )
    with importance_columns[1]:
        st.markdown("**Top importance chart**")
        st.bar_chart(importance_table.head(20).set_index("feature")["importance"])

    render_saved_model_prediction_tester(model_bundle)

    st.markdown("**Single-tree viewer**")
    viewer_columns = st.columns(2)
    tree_index_key = "saved_model_tree_index"
    if tree_index_key not in st.session_state:
        st.session_state[tree_index_key] = 0
    st.session_state[tree_index_key] = min(
        max(int(st.session_state[tree_index_key]), 0),
        len(model.estimators_) - 1,
    )
    with viewer_columns[0]:
        tree_index = int(
            st.number_input(
                "Tree index",
                min_value=0,
                max_value=len(model.estimators_) - 1,
                step=1,
                help=tooltip_text("tree_index"),
                key=tree_index_key,
            )
        )
    selected_tree_stats = (
        forest_stats.loc[forest_stats["tree_index"] == tree_index]
        .iloc[0]
    )
    selected_tree_max_depth = max(int(selected_tree_stats["depth"]), 0)
    tree_depth_key = "saved_model_tree_depth_limit"
    if tree_depth_key not in st.session_state:
        st.session_state[tree_depth_key] = min(4, selected_tree_max_depth)
    st.session_state[tree_depth_key] = min(
        max(int(st.session_state[tree_depth_key]), 0),
        selected_tree_max_depth,
    )

    with viewer_columns[1]:
        tree_depth_limit = int(
            st.number_input(
                "Tree depth to draw",
                min_value=0,
                max_value=selected_tree_max_depth,
                step=1,
                key=tree_depth_key,
                help=tooltip_text("tree_depth_limit"),
            )
        )

    st.caption(
        "Selected tree summary: "
        f"rank #{int(selected_tree_stats['importance_rank'])}, "
        f"weight {selected_tree_stats['weight']:.4f}, "
        f"importance score {selected_tree_stats['importance_score']:.4f}, "
        f"depth {int(selected_tree_stats['depth'])}, "
        f"leaves {int(selected_tree_stats['leaves'])}."
    )
    st.caption(
        "Use the drawing depth to control how many levels are shown. "
        "The tree canvas supports full-depth zooming with the buttons, mouse wheel, and drag-to-pan."
    )

    try:
        tree_svg_html = build_tree_svg_html(
            model,
            metadata=metadata,
            feature_names=feature_names,
            feature_labels=feature_labels,
            tree_index=tree_index,
            max_depth=tree_depth_limit,
            dark_mode=forest_mode_enabled(),
        )
        tree_frame_height = min(max(tree_depth_limit, 4) * 62 + 520, 1080)
        components.html(tree_svg_html, height=tree_frame_height, scrolling=False)
    except Exception as exc:
        st.warning(f"Could not render the tree graph: {exc}")

    st.markdown("**Human-readable rules for selected tree**")
    st.caption("Each branch starts collapsed. Expand only the paths you want to inspect.")
    st.markdown(
        build_tree_rules_html(
            model,
            metadata=metadata,
            feature_names=feature_names,
            feature_labels=feature_labels,
            tree_index=tree_index,
            max_depth=tree_depth_limit,
        ),
        unsafe_allow_html=True,
    )


def infer_problem_type(target: pd.Series) -> str:
    clean_target = target.dropna()
    if clean_target.empty:
        raise ValueError("The target column is empty after removing missing values.")

    if (
        pd.api.types.is_object_dtype(clean_target)
        or pd.api.types.is_string_dtype(clean_target)
        or isinstance(clean_target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(clean_target)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(clean_target):
        unique_count = clean_target.nunique()
        unique_ratio = unique_count / max(len(clean_target), 1)
        if unique_count <= 20 and unique_ratio <= 0.2:
            return "classification"

    return "regression"


@st.cache_data(show_spinner=False)
def convert_columns_to_dates(
    frame: pd.DataFrame,
    date_columns: list[str],
    *,
    dayfirst: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    converted = frame.copy()
    summary_rows: list[dict[str, int | str]] = []

    for column in date_columns:
        source = converted[column]
        source_non_null = int(source.notna().sum())
        if pd.api.types.is_datetime64_any_dtype(source) or isinstance(
            source.dtype,
            pd.DatetimeTZDtype,
        ):
            parsed = pd.to_datetime(source, errors="coerce")
        else:
            parsed = pd.to_datetime(
                source.astype("string"),
                errors="coerce",
                dayfirst=dayfirst,
            )

        parsed_non_null = int(parsed.notna().sum())
        converted[column] = parsed
        summary_rows.append(
            {
                "column": column,
                "parsed_rows": parsed_non_null,
                "invalid_rows": max(source_non_null - parsed_non_null, 0),
            }
        )

    return converted, pd.DataFrame(summary_rows)


def validate_custom_column_name(column_name: str, existing_columns: list[str]) -> str:
    cleaned_name = str(column_name).strip()
    if not cleaned_name:
        raise ValueError("Custom column name cannot be blank.")
    if cleaned_name in existing_columns:
        raise ValueError(f"`{cleaned_name}` already exists in the dataset.")
    return cleaned_name


def build_paired_text_column(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    *,
    separator: str = " | ",
) -> pd.Series:
    if left_column == right_column:
        raise ValueError("Choose two different columns for column pairing.")

    def combine_values(left_value: Any, right_value: Any) -> Any:
        parts = [
            str(value)
            for value in (left_value, right_value)
            if not pd.isna(value) and str(value) != ""
        ]
        if not parts:
            return pd.NA
        return separator.join(parts)

    return pd.Series(
        [
            combine_values(left_value, right_value)
            for left_value, right_value in zip(
                frame[left_column],
                frame[right_column],
                strict=False,
            )
        ],
        index=frame.index,
        dtype="object",
    )


def build_group_average_column(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> pd.Series:
    numeric_values = pd.to_numeric(frame[value_column], errors="coerce")
    return numeric_values.groupby(frame[group_column], dropna=False).transform("mean")


def get_custom_column_source_names(spec: dict[str, Any]) -> list[str]:
    if spec["kind"] == "pair":
        return [str(spec["left_column"]), str(spec["right_column"])]
    if spec["kind"] == "group_average":
        return [str(spec["value_column"]), str(spec["group_column"])]
    raise ValueError(f"Unsupported custom column type: {spec['kind']}")


def resolve_custom_column_input_dependencies(
    specs: list[dict[str, Any]],
) -> dict[str, set[str]]:
    dependency_map: dict[str, set[str]] = {}
    for spec in specs:
        resolved_sources: set[str] = set()
        for source_name in get_custom_column_source_names(spec):
            if source_name in dependency_map:
                resolved_sources.update(dependency_map[source_name])
            else:
                resolved_sources.add(source_name)
        dependency_map[str(spec["name"])] = resolved_sources
    return dependency_map


def get_required_custom_specs(
    specs: list[dict[str, Any]],
    selected_feature_names: list[str],
) -> list[dict[str, Any]]:
    required_names = {str(name) for name in selected_feature_names}
    required_specs: list[dict[str, Any]] = []

    for spec in reversed(specs):
        spec_name = str(spec["name"])
        if spec_name not in required_names:
            continue
        required_specs.append(spec)
        required_names.update(get_custom_column_source_names(spec))

    required_specs.reverse()
    return required_specs


def get_target_leaky_custom_features(
    specs: list[dict[str, Any]],
    target_column: str,
) -> list[str]:
    dependency_map = resolve_custom_column_input_dependencies(specs)
    target_name = str(target_column)
    return [
        feature_name
        for feature_name, dependencies in dependency_map.items()
        if target_name in dependencies
    ]


def get_required_input_feature_columns(
    base_columns: Sequence[Any],
    selected_feature_names: list[str],
    specs: list[dict[str, Any]],
) -> list[str]:
    dependency_map = resolve_custom_column_input_dependencies(specs)
    required_inputs: set[str] = set()
    for feature_name in selected_feature_names:
        feature_key = str(feature_name)
        if feature_key in dependency_map:
            required_inputs.update(dependency_map[feature_key])
        else:
            required_inputs.add(feature_key)
    return [str(column) for column in base_columns if str(column) in required_inputs]


def get_custom_column_schema_signature(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(str(column) for column in frame.columns)


def sync_custom_column_state(frame: pd.DataFrame) -> None:
    schema_signature = get_custom_column_schema_signature(frame)
    state_key = "custom_column_schema_signature"
    spec_key = "custom_column_specs"
    if st.session_state.get(state_key) != schema_signature:
        st.session_state[state_key] = schema_signature
        st.session_state[spec_key] = []
        st.session_state["pair_column_name_input"] = "paired_column_1"
        st.session_state["pair_column_separator_input"] = " | "
        st.session_state["group_average_name_input"] = "group_average_column_1"


def sync_selectbox_state(
    key: str,
    options: Sequence[Any],
    *,
    default: Any | None = None,
) -> None:
    option_list = list(options)
    if not option_list:
        st.session_state.pop(key, None)
        return

    current = st.session_state.get(key)
    if current in option_list:
        return

    if default in option_list:
        st.session_state[key] = default
        return

    st.session_state[key] = option_list[0]


def sync_multiselect_state(
    key: str,
    options: Sequence[Any],
    *,
    default: Sequence[Any] | None = None,
) -> None:
    option_list = list(options)
    current = st.session_state.get(key)
    if isinstance(current, tuple):
        current = list(current)

    if isinstance(current, list):
        filtered = [value for value in current if value in option_list]
        if filtered or not current:
            st.session_state[key] = filtered
            return

    default_values = list(default) if default is not None else []
    st.session_state[key] = [value for value in default_values if value in option_list]


@st.cache_data(show_spinner=False)
def apply_custom_column_specs(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, str]], list[str]]:
    derived_data = frame.copy()
    summary_rows: list[dict[str, str]] = []
    errors: list[str] = []

    for index, spec in enumerate(specs, start=1):
        try:
            new_column_name = validate_custom_column_name(
                spec["name"],
                list(derived_data.columns),
            )
            if spec["kind"] == "pair":
                derived_data[new_column_name] = build_paired_text_column(
                    derived_data,
                    str(spec["left_column"]),
                    str(spec["right_column"]),
                    separator=str(spec.get("separator", " | ")),
                )
                summary_rows.append(
                    {
                        "Spec index": str(index - 1),
                        "Step": str(index),
                        "New column": new_column_name,
                        "Feature type": "Column pairing",
                        "Definition": (
                            f"{spec['left_column']}{spec.get('separator', ' | ')}"
                            f"{spec['right_column']}"
                        ),
                    }
                )
            elif spec["kind"] == "group_average":
                derived_data[new_column_name] = build_group_average_column(
                    derived_data,
                    str(spec["value_column"]),
                    str(spec["group_column"]),
                )
                summary_rows.append(
                    {
                        "Spec index": str(index - 1),
                        "Step": str(index),
                        "New column": new_column_name,
                        "Feature type": "Average of X by Y",
                        "Definition": (
                            f"mean({spec['value_column']}) by {spec['group_column']}"
                        ),
                    }
                )
            else:
                raise ValueError(f"Unsupported custom column type: {spec['kind']}")
        except Exception as exc:
            errors.append(
                f"Custom column step {index} (`{spec.get('name', 'unnamed')}`) could not be applied: {exc}"
            )

    return derived_data, summary_rows, errors


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, specs: list[dict[str, Any]] | None = None):
        self.specs = specs or []

    def fit(self, X: pd.DataFrame, y: Any = None) -> CustomFeatureTransformer:
        frame = X.copy()
        self.fitted_specs_: list[dict[str, Any]] = []
        self.feature_names_in_ = np.array([str(column) for column in X.columns], dtype=object)

        for spec in self.specs:
            fitted_spec = dict(spec)
            new_column_name = validate_custom_column_name(
                str(spec["name"]),
                list(frame.columns),
            )
            fitted_spec["name"] = new_column_name

            if spec["kind"] == "pair":
                frame[new_column_name] = build_paired_text_column(
                    frame,
                    str(spec["left_column"]),
                    str(spec["right_column"]),
                    separator=str(spec.get("separator", " | ")),
                )
            elif spec["kind"] == "group_average":
                value_column = str(spec["value_column"])
                group_column = str(spec["group_column"])
                numeric_values = pd.to_numeric(frame[value_column], errors="coerce")
                fitted_spec["_group_means"] = numeric_values.groupby(
                    frame[group_column],
                    dropna=False,
                ).mean()
                frame[new_column_name] = build_group_average_column(
                    frame,
                    value_column,
                    group_column,
                )
            else:
                raise ValueError(f"Unsupported custom column type: {spec['kind']}")

            self.fitted_specs_.append(fitted_spec)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fitted_specs_"):
            raise ValueError("CustomFeatureTransformer must be fitted before transform.")

        frame = X.copy()
        for spec in self.fitted_specs_:
            new_column_name = str(spec["name"])
            if spec["kind"] == "pair":
                frame[new_column_name] = build_paired_text_column(
                    frame,
                    str(spec["left_column"]),
                    str(spec["right_column"]),
                    separator=str(spec.get("separator", " | ")),
                )
            elif spec["kind"] == "group_average":
                group_values = frame[str(spec["group_column"])]
                mapped_values = group_values.map(spec["_group_means"])
                frame[new_column_name] = pd.to_numeric(mapped_values, errors="coerce")
            else:
                raise ValueError(f"Unsupported custom column type: {spec['kind']}")

        return frame


class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> DatetimeFeatureTransformer:
        self.feature_names_in_ = np.array([str(column) for column in X.columns], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return normalize_feature_dtypes(X)


@st.cache_data(show_spinner=False)
def normalize_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    datetime_columns = normalized.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    for column in datetime_columns:
        series = normalized[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            series = series.dt.tz_localize(None)

        normalized[f"{column}__year"] = series.dt.year.astype("float64")
        normalized[f"{column}__month"] = series.dt.month.astype("float64")
        normalized[f"{column}__day"] = series.dt.day.astype("float64")
        normalized[f"{column}__day_of_week"] = series.dt.dayofweek.astype("float64")
        normalized[f"{column}__day_of_year"] = series.dt.dayofyear.astype("float64")
        normalized[f"{column}__quarter"] = series.dt.quarter.astype("float64")
        normalized[f"{column}__is_month_start"] = np.where(
            series.notna(),
            series.dt.is_month_start.astype(float),
            np.nan,
        )
        normalized[f"{column}__is_month_end"] = np.where(
            series.notna(),
            series.dt.is_month_end.astype(float),
            np.nan,
        )
        normalized[f"{column}__ordinal"] = series.map(
            lambda value: float(value.toordinal()) if pd.notna(value) else np.nan
        )

        # Keep the same schema when a held-out or manual input contains only
        # midnight timestamps or missing dates. Date-only values use zeros.
        normalized[f"{column}__hour"] = series.dt.hour.astype("float64")
        normalized[f"{column}__minute"] = series.dt.minute.astype("float64")
        normalized[f"{column}__second"] = series.dt.second.astype("float64")

    if datetime_columns:
        normalized = normalized.drop(columns=datetime_columns)
    return normalized


def build_input_feature_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    original_feature_names = [str(column) for column in frame.columns]
    schema: list[dict[str, Any]] = []

    for column in frame.columns:
        column_name = str(column)
        series = frame[column]
        options: list[Any] = []

        if pd.api.types.is_bool_dtype(series):
            kind = "categorical"
            options = [False, True]
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series) or isinstance(
            series.dtype,
            pd.DatetimeTZDtype,
        ):
            kind = "datetime"
        else:
            cleaned_options: list[Any] = []
            for value in series.dropna().unique().tolist():
                python_value = value.item() if isinstance(value, np.generic) else value
                if python_value not in cleaned_options:
                    cleaned_options.append(python_value)
            if 0 < len(cleaned_options) <= 50:
                kind = "categorical"
                options = cleaned_options
            else:
                kind = "text"

        schema.append(
            {
                "name": column_name,
                "label": humanize_feature_name(column_name, original_feature_names),
                "kind": kind,
                "options": options,
            }
        )

    return schema


def strip_datetime_timezone(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_localize(None)
    return series


def has_time_component(series: pd.Series) -> bool:
    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    return bool(
        (
            (non_null_values.dt.hour != 0)
            | (non_null_values.dt.minute != 0)
            | (non_null_values.dt.second != 0)
            | (non_null_values.dt.microsecond != 0)
            | (non_null_values.dt.nanosecond != 0)
        ).any()
    )


def encode_datetime_target(target: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    clean_target = strip_datetime_timezone(target)
    includes_time = has_time_component(clean_target)
    numeric_target = clean_target.astype("int64").astype("float64") / NANOSECONDS_PER_DAY
    target_details = {
        "is_datetime_target": True,
        "includes_time": includes_time,
        "metric_unit": "days",
        "display_format": "%Y-%m-%d %H:%M:%S" if includes_time else "%Y-%m-%d",
        "target_strategy": "timeline_regression",
    }
    return pd.Series(numeric_target, index=target.index, name=target.name), target_details


def decode_datetime_predictions(
    values: pd.Series | np.ndarray,
    *,
    includes_time: bool,
) -> pd.Series:
    numeric_values = pd.Series(values, dtype="float64")
    if not includes_time:
        numeric_values = numeric_values.round()
    decoded = pd.to_datetime(
        numeric_values,
        unit="D",
        origin="unix",
        errors="coerce",
    )
    if not includes_time:
        decoded = decoded.dt.normalize()
    return decoded


def format_datetime_output(values: pd.Series | np.ndarray, display_format: str) -> pd.Series:
    datetime_values = pd.to_datetime(values, errors="coerce")
    return datetime_values.dt.strftime(display_format).fillna("Invalid date")


def build_series_column_config(
    series: pd.Series,
    *,
    label: str,
    help_text: str,
) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return st.column_config.NumberColumn(label, help=help_text)
    return st.column_config.TextColumn(label, help=help_text)


def get_date_conversion_column_config() -> dict[str, Any]:
    return {
        "column": st.column_config.TextColumn(
            "Column",
            help=with_story_reference(
                "The source column selected for date parsing.",
                "The Setup. These are notebook fields you want the forest to read as time-aware clues.",
            ),
        ),
        "parsed_rows": st.column_config.NumberColumn(
            "Parsed rows",
            help=with_story_reference(
                "Number of non-empty rows that were successfully converted into valid dates.",
                "The Setup. This tells you how much of the field notebook was successfully translated into usable clues.",
            ),
            format="%d",
        ),
        "invalid_rows": st.column_config.NumberColumn(
            "Invalid rows",
            help=with_story_reference(
                "Number of non-empty rows that could not be interpreted as valid dates.",
                "The Setup. These are the notes the forest could not safely understand as dates.",
            ),
            format="%d",
        ),
    }


def get_metric_column_config(target_details: dict[str, Any]) -> dict[str, Any]:
    value_help = (
        "Metric value on the held-out test set. For date targets, the day-based errors are measured on the internal numeric timeline."
        if target_details["is_datetime_target"]
        else "Metric value on the held-out test set."
    )
    return {
        "Metric": st.column_config.TextColumn(
            "Metric",
            help=with_story_reference(
                "Name of the evaluation metric used to judge model performance.",
                "Out-of-Bag Error and other reality checks. This is how the app reports whether the forest learned the pattern or just memorized old stories.",
            ),
        ),
        "Value": st.column_config.NumberColumn(
            "Value",
            help=with_story_reference(
                value_help,
                "Out-of-Bag Error and other reality checks. This is the measurable score from the forest's reality check.",
            ),
        ),
    }


def get_prediction_column_config(
    prediction_frame: pd.DataFrame,
    *,
    target_details: dict[str, Any],
) -> dict[str, Any]:
    if target_details["is_datetime_target"]:
        return {
            "actual": st.column_config.TextColumn(
                "Actual",
                help=with_story_reference(
                    "The true target date from the test set.",
                    "The Forest in Action. This is what really happened in the story after the new example arrived.",
                ),
            ),
            "predicted": st.column_config.TextColumn(
                "Predicted",
                help=with_story_reference(
                    "The model's predicted target date after converting back from the internal numeric timeline.",
                    "Taking a Vote. This is the crowd's final answer after all the trees cast their votes.",
                ),
            ),
            "error_days": st.column_config.NumberColumn(
                "Error (days)",
                help=with_story_reference(
                    "Absolute difference between the true and predicted target dates, measured in days.",
                    "The Story of Errors. This shows how far the forest drifted from the true outcome on that example.",
                ),
            ),
        }

    return {
        "actual": build_series_column_config(
            prediction_frame["actual"],
            label="Actual",
            help_text=with_story_reference(
                "The true target value from the test set.",
                "The Forest in Action. This is what the mushroom really turned out to be.",
            ),
        ),
        "predicted": build_series_column_config(
            prediction_frame["predicted"],
            label="Predicted",
            help_text=with_story_reference(
                "The model's prediction for that test row.",
                "Taking a Vote. This is the forest's final answer after the trees cast their votes.",
            ),
        ),
    }


def get_feature_importance_column_config() -> dict[str, Any]:
    return {
        "feature": st.column_config.TextColumn(
            "Feature",
            help=with_story_reference(
                "Expanded feature name used by the trained model after preprocessing.",
                "Feature Importance. These are the clues the forest ended up questioning while it learned.",
            ),
        ),
        "importance": st.column_config.NumberColumn(
            "Importance",
            help=with_story_reference(
                "Relative contribution of the feature to the forest's split decisions. Higher values mean the model relied on it more.",
                "Feature Importance. This is how strongly that clue shaped the forest's thinking.",
            ),
        ),
    }


def get_confusion_matrix_column_config(frame: pd.DataFrame) -> dict[str, Any]:
    column_config: dict[str, Any] = {
        "Actual class": st.column_config.TextColumn(
            "Actual class",
            help=with_story_reference(
                "The true class label for the row.",
                "The Forest in Action. This is what the mushroom really was in the story.",
            ),
        )
    }
    for column in frame.columns:
        if column == "Actual class":
            continue
        predicted_label = str(column).replace("Predicted: ", "")
        column_config[column] = st.column_config.NumberColumn(
            column,
            help=with_story_reference(
                f"Count of rows from the actual class that the model predicted as {predicted_label}.",
                "Taking a Vote. This shows where the crowd agreed, hesitated, or voted for the wrong class.",
            ),
            format="%d",
        )
    return column_config


def get_tree_statistics_column_config() -> dict[str, Any]:
    return {
        "tree_index": st.column_config.NumberColumn(
            "Tree index",
            help="Index of the individual tree inside the forest.",
            format="%d",
        ),
        "importance_rank": st.column_config.NumberColumn(
            "Importance rank",
            help=(
                "Rank of the tree by total impurity reduction inside that tree. "
                "Rank 1 contributed the strongest split gains."
            ),
            format="%d",
        ),
        "weight": st.column_config.NumberColumn(
            "Weight",
            help=(
                "Relative share of total impurity reduction across the forest. "
                "This is a contribution weight, not a separate voting weight."
            ),
            format="%.4f",
        ),
        "importance_score": st.column_config.NumberColumn(
            "Importance score",
            help="Absolute impurity-reduction score for that tree before normalization.",
            format="%.4f",
        ),
        "depth": st.column_config.NumberColumn(
            "Depth",
            help="Maximum depth reached by the tree.",
            format="%d",
        ),
        "leaves": st.column_config.NumberColumn(
            "Leaves",
            help="Number of terminal leaves in the tree.",
            format="%d",
        ),
        "nodes": st.column_config.NumberColumn(
            "Nodes",
            help="Total node count in the tree, including splits and leaves.",
            format="%d",
        ),
    }


def update_training_progress(
    progress_bar: Any,
    status_placeholder: Any,
    value: int,
    message: str,
) -> None:
    progress_bar.progress(value)
    status_placeholder.caption(f"Training progress: {message}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were selected.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(problem_type: str, params: dict[str, Any]) -> RandomForestClassifier | RandomForestRegressor:
    common_args = {
        "n_estimators": params["n_estimators"],
        "criterion": params["criterion"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": params["max_features"],
        "bootstrap": params["bootstrap"],
        "random_state": params["random_state"],
        "n_jobs": -1,
    }
    if params["bootstrap"]:
        common_args["max_samples"] = params["max_samples"]

    if problem_type == "classification":
        common_args["class_weight"] = params["class_weight"]
        return RandomForestClassifier(**common_args)

    return RandomForestRegressor(**common_args)


def format_python_literal(value: Any) -> str:
    return pformat(value, width=88, sort_dicts=False)


def build_training_code_preview(
    *,
    feature_columns: list[str],
    target_column: str,
    input_feature_columns: list[str],
    required_custom_specs: list[dict[str, Any]],
    parameter_summary: dict[str, Any],
    problem_type: str,
    test_size: float,
    random_state: int,
    shuffle_rows: bool,
    stratify_split: bool,
    run_cv: bool,
    cv_folds: int,
    target_is_datetime: bool,
) -> str:
    lines = [
        "# `data` is the preview dataframe after date parsing and custom-column setup.",
        "# `base_data` is the dataframe used before pipeline-generated features are added.",
        "",
        f"feature_columns = {format_python_literal(feature_columns)}",
        f"target_column = {format_python_literal(target_column)}",
        f"required_custom_specs = {format_python_literal(required_custom_specs)}",
        f"input_feature_columns = {format_python_literal(input_feature_columns)}",
        "",
        "model_input = base_data[input_feature_columns + [target_column]].copy()",
        "model_input = model_input.dropna(subset=[target_column])",
        "",
        "y = model_input[target_column]",
        "raw_feature_input = model_input[input_feature_columns].copy()",
    ]

    if target_is_datetime:
        lines.extend(
            [
                'target_details = {"is_datetime_target": False}',
                "y, target_details = encode_datetime_target(y)",
            ]
        )

    lines.extend(
        [
            "",
            f"parameter_summary = {format_python_literal(parameter_summary)}",
            "",
            "preview_training_features = normalize_feature_dtypes(data[feature_columns])",
            "preprocessor = build_preprocessor(preview_training_features)",
            f"model = build_model({problem_type!r}, parameter_summary)",
            "",
            "pipeline_steps = []",
            "if required_custom_specs:",
            "    pipeline_steps.append(",
            '        ("custom_features", CustomFeatureTransformer(required_custom_specs))',
            "    )",
            "pipeline_steps.extend(",
            "    [",
            '        ("datetime_features", DatetimeFeatureTransformer()),',
            '        ("preprocessor", preprocessor),',
            '        ("model", model),',
            "    ]",
            ")",
            "pipeline = Pipeline(steps=pipeline_steps)",
            "",
            (
                "stratify_labels = y"
                if problem_type == "classification" and stratify_split
                else "stratify_labels = None"
            ),
            "X_train, X_test, y_train, y_test = train_test_split(",
            "    raw_feature_input,",
            "    y,",
            f"    test_size={test_size},",
            f"    random_state={random_state},",
            f"    shuffle={shuffle_rows},",
            "    stratify=stratify_labels,",
            ")",
            "",
            "pipeline.fit(X_train, y_train)",
            "predictions = pipeline.predict(X_test)",
        ]
    )

    if run_cv:
        lines.extend(
            [
                "",
                f"scoring = get_scoring({problem_type!r})",
                "cv_scores = cross_val_score(",
                "    pipeline,",
                "    X_train,",
                "    y_train,",
                f"    cv={cv_folds},",
                "    scoring=scoring,",
                "    n_jobs=1,",
                ")",
            ]
        )

    return "\n".join(lines)


@st.dialog("Training Code Used", width="large")
def render_training_code_dialog(code_preview: str) -> None:
    st.caption(
        "This mirrors the training path in the app with the current selections filled in."
    )
    st.code(code_preview, language="python")
    if st.button(
        "Close",
        use_container_width=True,
        key="close_training_code_dialog_button",
    ):
        st.rerun()


def get_metric_summary(problem_type: str, y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    if problem_type == "classification":
        metrics = {
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision (weighted)": precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "Recall (weighted)": recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "F1 (weighted)": f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
        metrics = {
            "MAE": mean_absolute_error(y_true, predictions),
            "RMSE": rmse,
            "R2": r2_score(y_true, predictions),
        }

    return pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    )


def get_datetime_target_metric_summary(y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    metrics = {
        "MAE (days)": mean_absolute_error(y_true, predictions),
        "RMSE (days)": rmse,
        "R2": r2_score(y_true, predictions),
    }
    return pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    )


def get_scoring(problem_type: str) -> str:
    return "f1_weighted" if problem_type == "classification" else "r2"


def get_feature_importance_table(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance_table.sort_values("importance", ascending=False)


def format_parameter_value(choice: str) -> str | None:
    mapping = {
        "sqrt": "sqrt",
        "log2": "log2",
        "All features": None,
    }
    return mapping[choice]


def initialize_training_state() -> None:
    defaults = {
        "training_date_columns": [],
        "training_date_dayfirst": False,
        "pair_column_name_input": "paired_column_1",
        "pair_column_separator_input": " | ",
        "group_average_name_input": "group_average_column_1",
        "problem_type_choice": "Auto-detect",
        "test_size_input": 0.2,
        "random_state_input": 42,
        "shuffle_rows_input": True,
        "stratify_split_input": True,
        "run_cv_input": False,
        "cv_folds_input": 5,
        "n_estimators_input": 200,
        "class_weight_choice": "None",
        "criterion_input": "gini",
        "use_max_depth_input": False,
        "max_depth_input": 10,
        "min_samples_split_input": 2,
        "min_samples_leaf_input": 1,
        "max_features_input": "sqrt",
        "bootstrap_input": True,
        "use_max_samples_input": False,
        "max_samples_input": 0.8,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main() -> None:
    st.set_page_config(
        page_title="Random Forest Trainer",
        page_icon=str(get_favicon_path()),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("forest_theme_enabled", False)
    st.session_state.setdefault("current_page", "main")
    st.session_state.setdefault("workflow_mode", "train")
    initialize_training_state()
    if not forest_mode_enabled() and st.session_state.get("current_page") == "story":
        st.session_state["current_page"] = "main"

    render_theme_mode()
    render_sidebar_restore()
    render_theme_toggle()
    render_story_navigation()

    if st.session_state.get("current_page") == "story":
        render_story_page()
        return
    if st.session_state.get("current_page") == "audit":
        render_audit_log_page()
        return

    header_columns = st.columns([4.3, 1.25, 1.35], gap="small")
    with header_columns[0]:
        st.title("Random Forest Trainer")
        st.caption(
            "Train a random forest from a CSV or inspect a saved forest artifact without editing code."
        )
    with header_columns[1]:
        st.write("")
        st.write("")
        st.button(
            "Audit Log",
            use_container_width=True,
            key="open_audit_log_button",
            help="Review and download the saved training audit log.",
            on_click=open_audit_page,
        )
    with header_columns[2]:
        st.write("")
        st.write("")
        if st.session_state.get("workflow_mode") == "inspect":
            st.button(
                "Back To Training",
                use_container_width=True,
                help=tooltip_text("workflow_mode"),
                on_click=open_training_view,
            )
        else:
            st.button(
                "Inspect Saved Model",
                use_container_width=True,
                help=tooltip_text("workflow_mode"),
                on_click=open_model_viewer,
            )

    if st.session_state.get("workflow_mode") == "inspect":
        render_saved_model_viewer()
        return

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help=tooltip_text("upload_csv"),
        key="training_upload_csv",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    try:
        data = load_csv(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    if data.empty:
        st.error("The uploaded CSV has no rows.")
        st.stop()

    st.subheader("Data Type Options")
    source_columns = list(data.columns)
    sync_multiselect_state("training_date_columns", source_columns, default=[])
    date_columns = st.multiselect(
        "Columns to convert to dates",
        options=source_columns,
        help=tooltip_text("date_columns"),
        key="training_date_columns",
    )
    date_dayfirst = st.checkbox(
        "Use day-first date parsing",
        help=tooltip_text("date_dayfirst"),
        key="training_date_dayfirst",
    )
    if date_columns:
        data, date_conversion_summary = convert_columns_to_dates(
            data,
            date_columns,
            dayfirst=date_dayfirst,
        )
        st.caption(
            "Date feature columns are expanded into numeric parts such as year, month, day, weekday, and ordinal date during training."
        )
        failed_date_columns = date_conversion_summary.loc[
            date_conversion_summary["parsed_rows"] == 0,
            "column",
        ].tolist()
        if failed_date_columns:
            st.warning(
                "These columns could not be parsed into dates: "
                + ", ".join(failed_date_columns)
            )
        with st.expander("Date conversion summary", expanded=False):
            st.caption("Hover the column headers for details.")
            st.dataframe(
                date_conversion_summary,
                use_container_width=True,
                hide_index=True,
                column_config=get_date_conversion_column_config(),
            )

    base_data = data
    st.subheader("Custom Columns")
    st.caption(
        "Create extra columns before preview and training. These derived columns are added to the uploaded dataset and can be selected as model features."
    )
    sync_custom_column_state(data)
    custom_column_specs = st.session_state.setdefault("custom_column_specs", [])
    derived_data, custom_column_rows, custom_column_errors = apply_custom_column_specs(
        data,
        custom_column_specs,
    )

    for error_message in custom_column_errors:
        st.error(error_message)

    if any(spec["kind"] == "group_average" for spec in custom_column_specs):
        st.info(
            "Grouped-average custom features are learned from the training split during fitting to avoid leakage. Preview values can differ slightly from held-out rows."
        )

    base_column_options = list(derived_data.columns)
    numeric_average_candidates = derived_data.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.setdefault(
        "pair_column_name_input",
        f"paired_column_{len(custom_column_specs) + 1}",
    )
    st.session_state.setdefault(
        "group_average_name_input",
        f"group_average_column_{len(custom_column_specs) + 1}",
    )
    if base_column_options:
        sync_selectbox_state(
            "pair_column_left_input",
            base_column_options,
            default=base_column_options[0],
        )
        sync_selectbox_state(
            "pair_column_right_input",
            base_column_options,
            default=base_column_options[1] if len(base_column_options) > 1 else base_column_options[0],
        )
        sync_selectbox_state(
            "group_average_group_input",
            base_column_options,
            default=base_column_options[0],
        )
    if numeric_average_candidates:
        sync_selectbox_state(
            "group_average_value_input",
            numeric_average_candidates,
            default=numeric_average_candidates[0],
        )

    custom_column_builders = st.columns(2, gap="large")
    with custom_column_builders[0]:
        with st.form("add_pair_column_form"):
            st.markdown("**Add paired text column**")
            pair_column_name = st.text_input(
                "Paired column name",
                help=tooltip_text("pair_column_name"),
                key="pair_column_name_input",
            )
            if base_column_options:
                pair_column_left = st.selectbox(
                    "First column",
                    options=base_column_options,
                    help=tooltip_text("pair_column_left"),
                    key="pair_column_left_input",
                )
                pair_column_right = st.selectbox(
                    "Second column",
                    options=base_column_options,
                    help=tooltip_text("pair_column_right"),
                    key="pair_column_right_input",
                )
            else:
                pair_column_left = ""
                pair_column_right = ""
                st.text_input(
                    "First column",
                    value="No columns available",
                    disabled=True,
                    help=tooltip_text("pair_column_left"),
                )
                st.text_input(
                    "Second column",
                    value="No columns available",
                    disabled=True,
                    help=tooltip_text("pair_column_right"),
                )
            pair_column_separator = st.text_input(
                "Separator",
                help=tooltip_text("pair_column_separator"),
                key="pair_column_separator_input",
            )
            add_pair_column = st.form_submit_button(
                "Add paired column",
                disabled=not base_column_options,
            )

        if add_pair_column:
            try:
                validate_custom_column_name(pair_column_name, list(derived_data.columns))
                build_paired_text_column(
                    derived_data,
                    pair_column_left,
                    pair_column_right,
                    separator=pair_column_separator,
                )
                custom_column_specs.append(
                    {
                        "kind": "pair",
                        "name": pair_column_name,
                        "left_column": pair_column_left,
                        "right_column": pair_column_right,
                        "separator": pair_column_separator,
                    }
                )
                st.session_state["pair_column_name_input"] = (
                    f"paired_column_{len(custom_column_specs) + 1}"
                )
                st.session_state["pair_column_separator_input"] = " | "
                st.rerun()
            except ValueError as exc:
                st.error(f"Pair column could not be added: {exc}")

    with custom_column_builders[1]:
        with st.form("add_group_average_column_form"):
            st.markdown("**Add grouped average column**")
            group_average_name = st.text_input(
                "Grouped average column name",
                help=tooltip_text("group_average_name"),
                key="group_average_name_input",
            )
            if numeric_average_candidates:
                group_average_value = st.selectbox(
                    "Average this numeric column",
                    options=numeric_average_candidates,
                    help=tooltip_text("group_average_value"),
                    key="group_average_value_input",
                )
            else:
                group_average_value = None
                st.text_input(
                    "Average this numeric column",
                    value="No numeric columns available",
                    disabled=True,
                    help=tooltip_text("group_average_value"),
                )
            if base_column_options:
                group_average_group = st.selectbox(
                    "Group by this column",
                    options=base_column_options,
                    help=tooltip_text("group_average_group"),
                    key="group_average_group_input",
                )
            else:
                group_average_group = ""
                st.text_input(
                    "Group by this column",
                    value="No columns available",
                    disabled=True,
                    help=tooltip_text("group_average_group"),
                )
            add_group_average = st.form_submit_button(
                "Add grouped average",
                disabled=not base_column_options or not numeric_average_candidates,
            )

        if not numeric_average_candidates:
            st.info("No numeric columns are available yet for the grouped-average feature.")

        if add_group_average:
            try:
                validate_custom_column_name(group_average_name, list(derived_data.columns))
                build_group_average_column(
                    derived_data,
                    str(group_average_value),
                    group_average_group,
                )
                custom_column_specs.append(
                    {
                        "kind": "group_average",
                        "name": group_average_name,
                        "value_column": str(group_average_value),
                        "group_column": group_average_group,
                    }
                )
                st.session_state["group_average_name_input"] = (
                    f"group_average_column_{len(custom_column_specs) + 1}"
                )
                st.rerun()
            except ValueError as exc:
                st.error(f"Grouped average column could not be added: {exc}")

    if custom_column_rows:
        st.dataframe(
            pd.DataFrame(custom_column_rows).drop(columns=["Spec index"]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Manage custom columns**")
        for index, summary_row in enumerate(custom_column_rows):
            row_columns = st.columns([5.5, 1.2], gap="small")
            with row_columns[0]:
                st.caption(
                    f"Step {summary_row['Step']}: `{summary_row['New column']}`"
                    f" using {summary_row['Feature type']} ({summary_row['Definition']})"
                )
            with row_columns[1]:
                if st.button("Remove", key=f"remove_custom_column_{index}", use_container_width=True):
                    custom_column_specs.pop(int(summary_row["Spec index"]))
                    st.rerun()

    if custom_column_specs:
        action_columns = st.columns([1.5, 5], gap="small")
        with action_columns[0]:
            if st.button("Clear All", key="clear_all_custom_columns", use_container_width=True):
                st.session_state["custom_column_specs"] = []
                st.session_state["pair_column_name_input"] = "paired_column_1"
                st.session_state["pair_column_separator_input"] = " | "
                st.session_state["group_average_name_input"] = "group_average_column_1"
                st.rerun()

    data = derived_data

    st.subheader("Dataset Overview")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Rows", f"{len(data):,}")
    overview_columns[1].metric("Columns", f"{len(data.columns):,}")
    overview_columns[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    overview_columns[3].metric(
        "Categorical columns",
        f"{len(data.select_dtypes(exclude=[np.number]).columns):,}",
    )

    with st.expander("Preview data", expanded=True):
        st.dataframe(data.head(25), use_container_width=True)

    selectable_columns = list(base_data.columns)
    sync_selectbox_state(
        "target_column_input",
        selectable_columns,
        default=selectable_columns[0] if selectable_columns else None,
    )
    target_column = st.selectbox(
        "Target column",
        options=selectable_columns,
        help=tooltip_text("target_column"),
        key="target_column_input",
    )
    target_is_datetime = pd.api.types.is_datetime64_any_dtype(base_data[target_column]) or isinstance(
        base_data[target_column].dtype,
        pd.DatetimeTZDtype,
    )
    blocked_custom_features = [
        feature_name
        for feature_name in get_target_leaky_custom_features(custom_column_specs, target_column)
        if feature_name in data.columns
    ]
    available_features = [
        column
        for column in data.columns
        if column != target_column and column not in blocked_custom_features
    ]

    if not available_features:
        st.error("The dataset needs at least one feature column in addition to the target.")
        st.stop()

    if blocked_custom_features:
        st.warning(
            "These custom features are unavailable for the selected target because they depend on the target column: "
            + ", ".join(blocked_custom_features)
        )

    sync_multiselect_state(
        "feature_columns_input",
        available_features,
        default=available_features,
    )
    feature_columns = st.multiselect(
        "Feature columns",
        options=available_features,
        help=tooltip_text("feature_columns"),
        key="feature_columns_input",
    )

    if not feature_columns:
        st.warning("Select at least one feature column to train the model.")
        st.stop()

    problem_type_choice = st.selectbox(
        "Problem type",
        options=["Auto-detect", "Classification", "Regression"],
        help=tooltip_text("problem_type"),
        key="problem_type_choice",
    )
    resolved_problem_type = (
        infer_problem_type(base_data[target_column])
        if problem_type_choice == "Auto-detect"
        else problem_type_choice.lower()
    )
    if target_is_datetime:
        if resolved_problem_type != "regression":
            st.warning(
                "Date target detected. The app is switching to regression and will predict against a numeric timeline under the hood."
            )
        resolved_problem_type = "regression"

    st.caption(f"Resolved model type: `{resolved_problem_type}`")
    if target_is_datetime:
        st.info(
            "Date target workaround enabled: the target will be converted to a numeric timeline for training, then predictions will be converted back into readable dates."
        )

    required_custom_specs = get_required_custom_specs(custom_column_specs, feature_columns)
    input_feature_columns = get_required_input_feature_columns(
        base_data.columns,
        feature_columns,
        required_custom_specs,
    )

    with st.sidebar:
        st.header("Train / Test Split")
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.5,
            step=0.05,
            help=tooltip_text("test_size"),
            key="test_size_input",
        )
        random_state = st.number_input(
            "Random state",
            min_value=0,
            step=1,
            help=tooltip_text("random_state"),
            key="random_state_input",
        )
        shuffle_rows = st.checkbox(
            "Shuffle rows",
            help=tooltip_text("shuffle_rows"),
            key="shuffle_rows_input",
        )
        stratify_split = st.checkbox(
            "Stratify classification split",
            disabled=resolved_problem_type != "classification" or not shuffle_rows,
            help=tooltip_text("stratify_split"),
            key="stratify_split_input",
        )
        run_cv = st.checkbox(
            "Run cross-validation",
            help=tooltip_text("run_cv"),
            key="run_cv_input",
        )
        cv_folds = st.slider(
            "CV folds",
            min_value=3,
            max_value=10,
            step=1,
            disabled=not run_cv,
            help=tooltip_text("cv_folds"),
            key="cv_folds_input",
        )

        st.header("Random Forest Parameters")
        n_estimators = st.slider(
            "n_estimators",
            min_value=50,
            max_value=1000,
            step=10,
            help=tooltip_text("n_estimators"),
            key="n_estimators_input",
        )

        if resolved_problem_type == "classification":
            criterion_options = ["gini", "entropy", "log_loss"]
            class_weight_choice = st.selectbox(
                "class_weight",
                options=["None", "balanced", "balanced_subsample"],
                help=tooltip_text("class_weight"),
                key="class_weight_choice",
            )
            criterion_help = tooltip_text("criterion_classification")
        else:
            criterion_options = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
            class_weight_choice = "None"
            criterion_help = tooltip_text("criterion_regression")

        sync_selectbox_state(
            "criterion_input",
            criterion_options,
            default=criterion_options[0],
        )
        criterion = st.selectbox(
            "criterion",
            options=criterion_options,
            help=criterion_help,
            key="criterion_input",
        )
        use_max_depth = st.checkbox(
            "Set max_depth",
            help=tooltip_text("use_max_depth"),
            key="use_max_depth_input",
        )
        max_depth = (
            int(
                st.number_input(
                    "max_depth",
                    min_value=1,
                    step=1,
                    help=tooltip_text("max_depth"),
                    key="max_depth_input",
                )
            )
            if use_max_depth
            else None
        )
        min_samples_split = int(
            st.number_input(
                "min_samples_split",
                min_value=2,
                step=1,
                help=tooltip_text("min_samples_split"),
                key="min_samples_split_input",
            )
        )
        min_samples_leaf = int(
            st.number_input(
                "min_samples_leaf",
                min_value=1,
                step=1,
                help=tooltip_text("min_samples_leaf"),
                key="min_samples_leaf_input",
            )
        )
        max_features = format_parameter_value(
            st.selectbox(
                "max_features",
                options=["sqrt", "log2", "All features"],
                help=tooltip_text("max_features"),
                key="max_features_input",
            )
        )
        bootstrap = st.checkbox(
            "bootstrap",
            help=tooltip_text("bootstrap"),
            key="bootstrap_input",
        )
        use_max_samples = st.checkbox(
            "Set max_samples",
            disabled=not bootstrap,
            help=tooltip_text("use_max_samples"),
            key="use_max_samples_input",
        )
        max_samples = (
            st.slider(
                "max_samples (fraction)",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                help=tooltip_text("max_samples"),
                key="max_samples_input",
            )
            if bootstrap and use_max_samples
            else None
        )

        parameter_summary = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "max_samples": max_samples,
            "class_weight": None if class_weight_choice == "None" else class_weight_choice,
            "random_state": int(random_state),
        }
        training_code_preview = build_training_code_preview(
            feature_columns=feature_columns,
            target_column=target_column,
            input_feature_columns=input_feature_columns,
            required_custom_specs=required_custom_specs,
            parameter_summary=parameter_summary,
            problem_type=resolved_problem_type,
            test_size=float(test_size),
            random_state=int(random_state),
            shuffle_rows=shuffle_rows,
            stratify_split=stratify_split,
            run_cv=run_cv,
            cv_folds=int(cv_folds),
            target_is_datetime=target_is_datetime,
        )
        if st.button(
            "Show Training Code",
            use_container_width=True,
            key="show_training_code_button",
        ):
            render_training_code_dialog(training_code_preview)

    model_input = base_data[input_feature_columns + [target_column]].copy()
    missing_target_rows = int(model_input[target_column].isna().sum())
    if missing_target_rows:
        st.warning(
            f"Dropping {missing_target_rows:,} row(s) with missing target values before training."
        )
    model_input = model_input.dropna(subset=[target_column])

    if len(model_input) < 5:
        st.error("There are not enough rows left to train after removing missing target values.")
        st.stop()

    y = model_input[target_column]
    raw_feature_input = model_input[input_feature_columns].copy()
    target_details: dict[str, Any] = {"is_datetime_target": False}
    if target_is_datetime:
        y, target_details = encode_datetime_target(y)

    if resolved_problem_type == "classification" and y.nunique() < 2:
        st.error("Classification requires at least two target classes.")
        st.stop()

    if resolved_problem_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        st.error("Regression requires a numeric target column, or choose Classification instead.")
        st.stop()

    if resolved_problem_type == "regression" and criterion == "poisson" and (y < 0).any():
        st.error("The `poisson` criterion requires a non-negative target column.")
        st.stop()

    if st.button("Train Model", type="primary", help=tooltip_text("train_model")):
        progress_bar = st.progress(0)
        progress_status = st.empty()
        try:
            update_training_progress(
                progress_bar,
                progress_status,
                10,
                "building preprocessing pipeline",
            )
            preview_training_features = normalize_feature_dtypes(data[feature_columns])
            input_feature_schema = build_input_feature_schema(
                model_input[input_feature_columns]
            )
            preprocessor = build_preprocessor(preview_training_features)
            model = build_model(resolved_problem_type, parameter_summary)
            pipeline_steps: list[tuple[str, Any]] = []
            if required_custom_specs:
                pipeline_steps.append(
                    ("custom_features", CustomFeatureTransformer(required_custom_specs))
                )
            pipeline_steps.extend(
                [
                    ("datetime_features", DatetimeFeatureTransformer()),
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            pipeline = Pipeline(steps=pipeline_steps)

            update_training_progress(
                progress_bar,
                progress_status,
                25,
                "splitting data into train and test sets",
            )
            stratify_labels = (
                y if resolved_problem_type == "classification" and stratify_split else None
            )
            X_train, X_test, y_train, y_test = train_test_split(
                raw_feature_input,
                y,
                test_size=test_size,
                random_state=int(random_state),
                shuffle=shuffle_rows,
                stratify=stratify_labels,
            )

            update_training_progress(
                progress_bar,
                progress_status,
                45,
                "training the random forest model",
            )
            pipeline.fit(X_train, y_train)

            update_training_progress(
                progress_bar,
                progress_status,
                60,
                "generating predictions",
            )
            predictions = pipeline.predict(X_test)

            update_training_progress(
                progress_bar,
                progress_status,
                72,
                "calculating evaluation metrics",
            )
            st.subheader("Results")
            st.caption("Hover the result table headers for details.")
            metric_table = (
                get_datetime_target_metric_summary(y_test, predictions)
                if target_details["is_datetime_target"]
                else get_metric_summary(resolved_problem_type, y_test, predictions)
            )
            st.dataframe(
                metric_table,
                use_container_width=True,
                hide_index=True,
                column_config=get_metric_column_config(target_details),
            )
            if target_details["is_datetime_target"]:
                st.caption(
                    "Date target predictions are trained on a numeric timeline internally and converted back into readable dates here."
                )

            cv_summary = {
                "enabled": False,
                "folds": None,
                "scoring": None,
                "mean": None,
                "std": None,
            }
            if run_cv:
                update_training_progress(
                    progress_bar,
                    progress_status,
                    82,
                    f"running {cv_folds}-fold cross-validation",
                )
                scoring = get_scoring(resolved_problem_type)
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=1,
                )
                cv_summary = {
                    "enabled": True,
                    "folds": int(cv_folds),
                    "scoring": scoring,
                    "mean": float(cv_scores.mean()),
                    "std": float(cv_scores.std()),
                }
                st.write(
                    f"Cross-validation ({cv_folds} folds, {scoring}): "
                    f"`{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}`"
                )

            update_training_progress(
                progress_bar,
                progress_status,
                90,
                "building result tables and charts",
            )
            result_columns = st.columns(2)

            with result_columns[0]:
                st.markdown("**Predictions sample**")
                if target_details["is_datetime_target"]:
                    actual_dates = decode_datetime_predictions(
                        y_test,
                        includes_time=target_details["includes_time"],
                    )
                    predicted_dates = decode_datetime_predictions(
                        predictions,
                        includes_time=target_details["includes_time"],
                    )
                    prediction_frame = pd.DataFrame(
                        {
                            "actual": format_datetime_output(
                                actual_dates,
                                target_details["display_format"],
                            ).reset_index(drop=True),
                            "predicted": format_datetime_output(
                                predicted_dates,
                                target_details["display_format"],
                            ).reset_index(drop=True),
                            "error_days": (
                                pd.Series(predictions).reset_index(drop=True)
                                - y_test.reset_index(drop=True)
                            ).abs().round(3),
                        }
                    )
                else:
                    prediction_frame = pd.DataFrame(
                        {
                            "actual": y_test.reset_index(drop=True),
                            "predicted": pd.Series(predictions),
                        }
                    )
                st.dataframe(
                    prediction_frame.head(25),
                    use_container_width=True,
                    column_config=get_prediction_column_config(
                        prediction_frame,
                        target_details=target_details,
                    ),
                )

            with result_columns[1]:
                importance_table = get_feature_importance_table(pipeline)
                st.markdown("**Top feature importances**")
                st.dataframe(
                    importance_table.head(25),
                    use_container_width=True,
                    hide_index=True,
                    column_config=get_feature_importance_column_config(),
                )
                st.bar_chart(
                    importance_table.head(15).set_index("feature")["importance"]
                )

            if resolved_problem_type == "classification":
                unique_classes = y.nunique()
                if unique_classes <= 25:
                    labels = pd.Index(pd.unique(pd.concat([y_test, pd.Series(predictions)])))
                    matrix = confusion_matrix(y_test, predictions, labels=labels)
                    confusion_frame = pd.DataFrame(
                        matrix,
                        index=[f"Actual: {label}" for label in labels],
                        columns=[f"Predicted: {label}" for label in labels],
                    )
                    st.markdown("**Confusion matrix**")
                    confusion_frame_display = confusion_frame.reset_index().rename(
                        columns={"index": "Actual class"}
                    )
                    st.dataframe(
                        confusion_frame_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config=get_confusion_matrix_column_config(
                            confusion_frame_display
                        ),
                    )
                else:
                    st.info("Confusion matrix skipped because the target has more than 25 classes.")

            update_training_progress(
                progress_bar,
                progress_status,
                94,
                "writing the training audit log",
            )
            audit_log_entry = build_training_audit_entry(
                problem_type=resolved_problem_type,
                target_column=target_column,
                feature_columns=feature_columns,
                input_feature_columns=input_feature_columns,
                input_feature_schema=input_feature_schema,
                date_columns=date_columns,
                date_dayfirst=bool(date_dayfirst),
                custom_column_specs=custom_column_specs,
                required_custom_specs=required_custom_specs,
                parameter_summary=parameter_summary,
                target_details=target_details,
                metric_table=metric_table,
                model_input_row_count=len(model_input),
                missing_target_rows=missing_target_rows,
                train_row_count=len(X_train),
                test_row_count=len(X_test),
                test_size=float(test_size),
                random_state=int(random_state),
                shuffle_rows=shuffle_rows,
                stratify_split=stratify_split,
                cv_summary=cv_summary,
            )
            try:
                audit_store_path = append_training_audit_log(audit_log_entry)
            except Exception as audit_exc:
                st.warning(f"Model trained, but the audit history could not be written: {audit_exc}")
            else:
                st.caption(f"Audit history saved to `{audit_store_path}`.")

            update_training_progress(
                progress_bar,
                progress_status,
                97,
                "packaging the trained model for download",
            )
            artifact = {
                "pipeline": pipeline,
                "problem_type": resolved_problem_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "input_feature_columns": input_feature_columns,
                "input_feature_schema": input_feature_schema,
                "parameters": parameter_summary,
                "target_details": target_details,
                "audit_log_entry": audit_log_entry,
            }
            st.download_button(
                label="Download trained model",
                data=pickle.dumps(artifact),
                file_name="random_forest_model.pkl",
                mime="application/octet-stream",
            )
            update_training_progress(
                progress_bar,
                progress_status,
                100,
                "complete",
            )
        except Exception as exc:
            progress_status.empty()
            st.error(f"Training failed: {exc}")


if __name__ == "__main__":
    main()

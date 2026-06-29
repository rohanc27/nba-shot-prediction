"""Shot profile feature engineering.

Creates combined shot-profile categories from location and action.
"""


def add_shot_profile_features(df):
    df = df.copy()

    df["shot_profile"] = (
        df["BASIC_ZONE"].astype(str)
        + " | "
        + df["ZONE_RANGE"].astype(str)
        + " | "
        + df["action_category"].astype(str)
    )

    df["zone_range"] = (
        df["ZONE_NAME"].astype(str)
        + " | "
        + df["ZONE_RANGE"].astype(str)
    )

    return df

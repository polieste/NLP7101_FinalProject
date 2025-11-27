import numpy as np
import pandas as pd


def split_user(
    group: pd.DataFrame,
    rng: np.random.Generator,
    val_users: set,
    test_users: set,
    tr: float,
    vr: tuple[float, float],
    mt: int,
) -> pd.DataFrame:
    uid = group["user_id"].iloc[0]
    n = len(group)

    if uid in val_users:
        group = group.copy()
        group["split"] = "val_unseen"
        return group

    if uid in test_users:
        group = group.copy()
        group["split"] = "test_unseen"
        return group

    train_end = max(1, int(np.floor(n * tr)))
    remain = n - train_end

    val_ratio = float(rng.uniform(vr[0], vr[1]))
    val_len = int(np.floor(remain * val_ratio))
    val_end = train_end + val_len

    if n - val_end < mt:
        val_end = max(train_end, n - mt)

    splits: list[str] = []
    splits.extend(["train"] * train_end)
    splits.extend(["val_seen"] * max(0, val_end - train_end))
    splits.extend(["test_seen"] * (n - val_end))

    group = group.copy()
    group["split"] = splits
    return group



def split_data(
    df: pd.DataFrame,
    seed: int = 11,
    u_val: float = 0.10,
    u_test: float = 0.15,
    tr: float = 0.75,
    vr: tuple[float, float] = (0.4, 0.6),
    ms: int = 3,
    mt: int = 1,
) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)

    user_ids = df["user_id"].unique()
    user_counts = df["user_id"].value_counts()

    small_users = set(user_counts[user_counts < ms].index)
    n_users = len(user_ids)
    n_val_users = max(1, int(round(n_users * u_val)))
    n_test_users = max(1, int(round(n_users * u_test)))

    candidate_users = [u for u in user_ids if u not in small_users]
    rng.shuffle(candidate_users)

    user_pool: list = list(small_users) + candidate_users

    val_users = set(user_pool[:n_val_users])
    test_users = set(user_pool[n_val_users:n_val_users + n_test_users])

    df = df.groupby("user_id", group_keys=False).apply(
        lambda g: split_user(g, rng, val_users, test_users, tr, vr, mt)
    )

    df["group"] = np.where(
        df["split"].isin(["val_unseen", "test_unseen"]),
        "unseen",
        "seen",
    )

    return df


if __name__ == '__main__':
    df = pd.read_csv('./TRAIN_RELEASE_3SEP2025/train_subtask1.csv')
    df = split_data(df)

    train = df[df.split == 'train']
    val = df[df.split.isin(['val_seen','val_unseen'])]
    test = df[df.split.isin(['test_seen','test_unseen'])]

    train.to_csv("split/train.csv", index=False)
    val.to_csv("split/val.csv", index=False)
    test.to_csv("split/test.csv", index=False)
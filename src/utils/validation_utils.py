import wandb
from sklearn.model_selection import KFold


def loso_validation(dataset, cfg, group_name, run_fn):
    loo_validation(dataset, cfg, "ID", group_name, run_fn)


def loto_validation(dataset, cfg, group_name, run_fn):
    loo_validation(dataset, cfg, "Class", group_name, run_fn)


def loo_validation(dataset, cfg, out: str, group_name, run_fn):
    ids = dataset.indexes[out].unique()
    for id_ in ids:
        train_idxs = [i for i, val in enumerate((dataset.indexes[out] != id_).to_list()) if val]
        test_idxs = [i for i, val in enumerate((dataset.indexes[out] == id_).to_list()) if val]

        run_fn(cfg, dataset, train_idxs, test_idxs, id_, group_name)


def kfold_validation(dataset, cfg, group_name, run_fn):
    k_fold = KFold(n_splits=cfg.training.folds, shuffle=True)
    for fold, (train_idxs, test_idxs) in enumerate(k_fold.split(dataset)):
        wandb.run.tags = [f"fold_{fold}"]
        run_fn(cfg, dataset, train_idxs, test_idxs, fold, group_name)

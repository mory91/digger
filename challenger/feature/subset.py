from constants import ALL_FEATURES, NSDI_FEATURES, FS_FEATURES, BEST_FEATURES


class FeatureSubset:
    def __init__(self, features=None):
        self.features = features

    def __call__(self, ds):
        if self.features is None:
            return ds.copy()
        return ds[self.features].copy()


get_nsdi_features = FeatureSubset(NSDI_FEATURES)
get_all_features = FeatureSubset(ALL_FEATURES)
get_fs_features = FeatureSubset(FS_FEATURES)
get_best_features = FeatureSubset(BEST_FEATURES)
get_full_features = FeatureSubset()

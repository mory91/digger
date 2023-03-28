from constants import ALL_FEATURES, NSDI_FEATURES, FS_FEATURES


class FeatureSubset:
    def __init__(self, features):
        self.features = features

    def __call__(self, ds):
        return ds[self.features].copy()


get_nsdi_features = FeatureSubset(NSDI_FEATURES)
get_all_features = FeatureSubset(ALL_FEATURES)
get_fs_features = FeatureSubset(FS_FEATURES)

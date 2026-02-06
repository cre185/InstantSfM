import importlib

# options that do not vary with feature_name
GENERAL_OPTIONS = {
    'skip_preprocessing': False,
    'skip_view_graph_calibration': False,
    'num_iteration_bundle_adjustment': 3,
    'skip_retriangulation': True,
    'num_iteration_retriangulation': 1,
    'skip_pruning': True,
}
RUNTIME_OPTIONS = {
    # below are all determined by data availability
    'use_depths': False,
    'use_semantic_filtering': False,
    'multi_camera_rig': False,
    'use_fixed_rel_poses': False,
}

class Config:
    def __init__(self, feature_name, manual_config_name=None):
        self.feature_name = feature_name

        config_module_names = {
            'colmap': 'instantsfm.config.colmap',
        }
        config_modules = {name: path for name, path in config_module_names.items()}
        
        if manual_config_name is not None:
            config_module_name = 'instantsfm.config.' + manual_config_name
        elif feature_name in config_module_names:
            config_module_name = config_modules[feature_name]
        else:
            raise ValueError('Invalid feature_name')
        
        config_module = importlib.import_module(config_module_name)
        CONFIG = getattr(config_module, 'CONFIG')

        self.OPTIONS = GENERAL_OPTIONS
        self.RUNTIME_OPTIONS = RUNTIME_OPTIONS
        self.VIEW_GRAPH_CALIBRATOR_OPTIONS = CONFIG['VIEW_GRAPH_CALIBRATOR_OPTIONS']
        self.INLIER_THRESHOLD_OPTIONS = CONFIG['INLIER_THRESHOLD_OPTIONS']
        self.SEMANTIC_OPTIONS = CONFIG['SEMANTIC_OPTIONS']
        self.ROTATION_ESTIMATOR_OPTIONS = CONFIG['ROTATION_ESTIMATOR_OPTIONS']
        self.L1_SOLVER_OPTIONS = CONFIG['L1_SOLVER_OPTIONS']
        self.TRACK_ESTABLISHMENT_OPTIONS = CONFIG['TRACK_ESTABLISHMENT_OPTIONS']
        self.GLOBAL_POSITIONER_OPTIONS = CONFIG['GLOBAL_POSITIONER_OPTIONS']
        self.BUNDLE_ADJUSTER_OPTIONS = CONFIG['BUNDLE_ADJUSTER_OPTIONS']
        self.TRIANGULATOR_OPTIONS = CONFIG['TRIANGULATOR_OPTIONS']
        self.VOTING_OPTIONS = CONFIG['VOTING_OPTIONS']
        self.FEATURE_HANDLER_OPTIONS = CONFIG['FEATURE_HANDLER_OPTIONS']
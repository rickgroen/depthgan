from .base_options import BaseOptions, boolstr


class EvaluateOptions(BaseOptions):

    def __init__(self):
        super(BaseOptions).__init__()

    def get_arguments(self, parser):
        parser = BaseOptions.get_arguments(self, parser)

        parser.add_argument('--predicted_disp_path',    type=str,       help='path to estimated disparities', required=True)
        parser.add_argument('--evaluate_mode',          type=str,       help='evaluation on either test or val set', default='test')

        parser.add_argument('--min_depth',              type=float,     help='minimum depth for evaluation', default=1e-3)
        parser.add_argument('--max_depth',              type=float,     help='maximum depth for evaluation', default=80)
        parser.add_argument('--eigen_crop',             type=boolstr,   help='if set, crops according to Eigen NIPS14', default=False)
        parser.add_argument('--garg_crop',              type=boolstr,   help='if set, crops according to Garg  ECCV16', default=True)

        return parser

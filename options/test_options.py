from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
    

class SlideInferOptions(BaseOptions):
    """This class includes extra options for slide inference.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')
        parser.add_argument('--masks_dir', type=str, default='otsu', help='path to the directory containing the masks, none, or "otsu".')
        parser.add_argument('--var_thresh', type=int, default=0, help='variance threshold for skipping patches.')
        parser.add_argument('--model_resolution', type=float, default=0, help='resolution to use for slide inference in mpp. default 0 uses baseline.')
        parser.add_argument('--save_resolution', type=float, default=0, help='resolution to construct slide at in mpp. default 0 uses baseline.')
        parser.add_argument('--stride', type=int, default=0, help='stride to use for slide inference in pixels. default 0 uses patch size.')
        parser.add_argument('--bkgrnd_heuristic', type=str, default='none', help='heuristic to use for background removal. none or morph.')
        parser.add_argument('--names', type=str, nargs='+', help='list of slide stems to run inference on.')
        parser.add_argument('--save_suffix', type=str, default='_proc', help='suffix to append to saved slide stem.')
        parser.add_argument('--valid_check', action='store_true', help='check if patches are valid before inference. Useful if getting tissue mask is hard.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

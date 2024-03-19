"""
BAD-Gaussians dataparser configs.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from bad_gaussians.deblur_nerf_dataparser import DeblurNerfDataParserConfig
from bad_gaussians.image_restoration_dataparser import ImageRestorationDataParserConfig

DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
ImageRestoreDataParser = DataParserSpecification(config=ImageRestorationDataParserConfig())

"""
BAD-Gaussians dataparser configs.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from bad_gaussians.deblur_nerf_dataparser import DeblurNerfDataParserConfig

DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())

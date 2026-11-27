from .ablation import build_final_model

VARIANTS=['baseline','ctrl_conv','cfpg_periph_only','cfpg_plus_csa','cfpg_plus_entropy_no_grn','cfpg']
def build_model(variant='cfpg',augment=False):
    if variant in VARIANTS:
        return build_final_model({'variant':variant},augment)
    if augment:
        raise ValueError('Baseline definitions are provided for inference; use their upstream training recipe.')
    if variant=='unet':
        from .baselines.unet_wrapper import build_unet
        return build_unet()
    if variant=='bisenetv2':
        from .baselines.bisenetv2 import BiSeNetV2
        return BiSeNetV2(num_classes=2,aux_loss=False,in_channels=3)
    if variant=='ddrnet23slim':
        from .baselines.ddrnet import build_ddrnet23_slim
        return build_ddrnet23_slim()
    if variant=='segformer_b0':
        from .baselines.segformer import build_segformer_b0
        return build_segformer_b0(num_classes=2)
    if variant=='ppliteseg_stdc1':
        from .baselines.pp_liteseg import build_pp_liteseg_stdc1
        return build_pp_liteseg_stdc1()
    if variant=='stdcseg':
        from .baselines.stdc_stages import BiSeNet
        from .baselines.stdc_inference import STDCMainInference
        from torch import nn
        wrapper=nn.Module();wrapper.model=BiSeNet('STDCNet813',2,pretrain_model='')
        return STDCMainInference(wrapper)
    raise ValueError(variant)

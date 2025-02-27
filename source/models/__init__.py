from .diffusion_model_text import SPLDiffusionModel

def load_model(cfg):
    if cfg['ModelType'] == 'SPLDiffusionModel':
        model = SPLDiffusionModel(
            diffusion_config=cfg['DiffusionConfig'],
            denoise_config=cfg['DenoiseConfig'],
            loss_weight=cfg['LossWeight'],
        )
    else:
        raise NotImplementedError

    return model

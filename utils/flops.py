from ptflops import get_model_complexity_info


def count_flops(model, opts, logger):
    macs, params = get_model_complexity_info(model, (3, opts.crop_size, opts.crop_size), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    return float(macs[:-5]), float(params[:-2])
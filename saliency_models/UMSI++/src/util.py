# from multiduration_models import md_sem, xception_se_lstm, sam_resnet_3d, xception_3d, xception_se_lstm_nodecoder
from singleduration_models import sam_resnet_new, UMSI, sam_resnet_nopriors
from losses_keras2 import loss_wrapper, kl_time, cc_time, nss_time, cc_match, kl_cc_combined, nss, kl_cc_nss_combined_new

MODELS = {
    # 'md_sem': (md_sem, 'singlestream'),
    # 'xception_se_lstm': (xception_se_lstm, 'singlestream'),
    # 'xception_3d': (xception_3d, 'singlestream'),
    # 'xception_se_lstm_nodecoder': (xception_se_lstm_nodecoder, 'singlestram'),
    # 'sam_resnet_3d': (sam_resnet_3d, 'singlestream'),
    'sam-resnet': (sam_resnet_new, 'simple'),
    'sam-noprior': (sam_resnet_nopriors, 'simple'),
    "UMSI": (UMSI, "simple")
}

LOSSES = {
    'kl': (kl_time, 'heatmap'),
    'cc': (cc_time, 'heatmap'),
    'nss': (nss, 'heatmap'),
    'ccmatch': (cc_match, 'heatmap'),
    "kl+cc": (kl_cc_combined, "heatmap"),
    "ours": (kl_cc_nss_combined_new, "heatmap")
}

def get_model_by_name(name): 
    """ Returns a model and a string indicating its mode of use."""
    if name not in MODELS: 
        allowed_models = list(MODELS.keys())
        raise RuntimeError("Model %s is not recognized. Please choose one of: %s" % (name, ",".join(allowed_models)))
    else: 
        return MODELS[name]

def get_loss_by_name(name, out_size): 
    """Gets the loss associated with a certain name. 

    If there is no custom loss associated with name `name`, returns the string
    `name` so that keras can interpret it as a keras loss.
    """
    if name not in LOSSES: 
        print("WARNING: found no custom loss with name %s, defaulting to a string." % name)
        return name, 'heatmap'
    else: 
        loss, out_type = LOSSES[name]
        loss = loss_wrapper(loss, out_size)
        return loss, out_type

def create_losses(loss_dict, out_size): 
    """Given a dictionary that maps loss names to weights, returns loss functions and weights in the correct order. 

    By convention, losses that take in a heatmap (as opposed to a fixmap) come first in the array of losses. This function enforces that convention.

    This function looks up the correct loss function by name and outputs the correct functions, ordering, and weights to pass to the model/generator.
    """
    l_hm = []
    l_hm_w = [] 
    l_fm = []
    l_fm_w = []
    lstr = ""
    for lname, wt in loss_dict.items(): 
        loss, out_type = get_loss_by_name(lname, out_size)    
        if out_type == 'heatmap': 
            l_hm.append(loss)
            l_hm_w.append(wt)
        else: 
            l_fm.append(loss)
            l_fm_w.append(wt)
        lstr += lname + str(wt)

    l = l_hm + l_fm
    lw = l_hm_w + l_fm_w
    n_heatmaps = len(l_hm)
    return l, lw, lstr, n_heatmaps    

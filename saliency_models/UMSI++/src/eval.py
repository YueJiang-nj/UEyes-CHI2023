import numpy as np
from tqdm import tqdm_notebook as tqdm
import tqdm
import scipy.ndimage
import matplotlib.pyplot as plt
#import scipy.stats
from keras.utils import Sequence
from eval_saliconeval import *

CODECHARTS_SUB_DATASETS = ["OOC", "SALICON", "STANFORD-ACTIONS", "CAT2000", "CROWD", "LAMEM"]
def name_to_dataset(name):
    if "CAT2000" in name or "LowRes" in name:
        d = "CAT2000"
    elif "COCO_" in name:
        d = "SALICON"
    elif "Crowd" in name:
        d = "CROWD"
    elif "LaMem" in name:
        d = "LAMEM"
    elif "OOC" in name:
        d = "OOC"
    elif "StanfordWritingActions" in name:
        d = "STANFORD-ACTIONS"
    else:
        raise Exception("Dataset could not be determined for name %s" % name)
    return d


def visualize_samples(model, gen, times):
    images, maps = gen.__getitem__(np.random.randint(len(gen_val)))

    preds = model.predict(images)

    times = [500, 3000, 5000]
    n_times = len(preds)
    assert len(times) == n_times
    batch_sz = len(preds[0])
    copy=0

    for batch in range(batch_sz):
        plt.imshow(reverse_preprocess(images[batch]))
        plt.title("original image %d" % batch)
        plt.show()

        plt.figure(figsize=[16, 10])
        n_row=n_times
        n_col=2

        for time in range(n_times):

            plt.subplot(n_row,n_col,time*n_col+1)
            plt.imshow(maps[time][batch][copy][:, :, 0])
            plt.title('Gt %dms' % times[time])

            plt.subplot(n_row,n_col,time*n_col+2)
            plt.imshow(preds[time][batch][copy][:, :, 0])
            plt.title('Prediction %dms' % times[time])

        plt.show()


def rmse(gr_truth, predicted):
    errors = gr_truth - predicted
    errors = errors**2
    rmse = np.sqrt(np.mean(errors))

    return rmse

def r2(gr_truth, predicted):

    truth_mean = np.mean(gr_truth)

    ssres = np.sum((predicted - gr_truth)**2)
    sstot = np.sum((gr_truth - truth_mean)**2)

    return 1 - ssres/sstot

def cc_npy(gt, predicted):
    M1 = np.divide(predicted - np.mean(predicted), np.std(predicted))
    M2 = np.divide(gt - np.mean(gt), np.std(gt))
    ret = np.corrcoef(M1.reshape(-1),M2.reshape(-1))[0][1]
    return ret

def nss_npy(gt_locs, predicted_map):
    assert gt_locs.shape == predicted_map.shape, \
    'dim missmatch in nss_npy: %s vs %s' % (gt_locs.shape, predicted_map.shape)
    predicted_map_norm = (predicted_map - np.mean(predicted_map))/np.std(predicted_map)
    dot = predicted_map_norm * gt_locs
    N = np.sum(gt_locs)
    ret = np.sum(dot)/N
    return ret

def kl_npy(gt, predicted):
    predicted = predicted/np.max(predicted)
    gt = gt/np.sum(gt)
    predicted = predicted/np.sum(predicted)
    kl_tensor = gt * np.log(gt / (predicted+1e-7) +1e-7)
    return np.sum(kl_tensor)

def sim_npy(gt, predicted):
    # Sum of min between distributions at each pixel
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))
    gt = gt/np.sum(gt)
    predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))
    predicted = predicted/np.sum(predicted)
    diff = np.minimum(gt, predicted)
    return np.sum(diff)

def emd_npy(gt, predicted):
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))
    gt = gt/np.sum(gt)
    predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))
    predicted = predicted/np.sum(predicted)

    gt_flat = gt.flatten()
    pred_flat = predicted.flatten()

    return scipy.stats.wasserstein_distance(gt_flat, pred_flat)

def cc_match_npy(gt, pred):
    ccm=0
    for t in range(len(gt)-1):
        ccm += np.abs(np.corrcoef(gt[t].reshape(-1), gt[t+1].reshape(-1))[0,1] - np.corrcoef(pred[t].reshape(-1), pred[t+1].reshape(-1))[0,1])
    return ccm / (len(gt)-1)

def acc(gt, pred):
    return np.argmax(gt) == np.argmax(pred)

def acc_per_class(gt, pred):
    ret = np.full((len(gt),),None)
    ret[np.argmax(gt)] = np.argmax(gt) == np.argmax(pred)
    return ret


def predict_and_save(model, test_img, inp_size, savedir, mode='multistream_concat', blur=False, test_img_base_path="", ext="png"):
    # if test_img_base_path is specified, then preserves the original
    # nested structure of the directory from which the stuff is pulled
    c=0
    if blur:
        print('BLURRING PREDICTIONS')
        if 'blur' not in savedir:
            savedir = savedir+'_blur'
    else:
        print('NOT BLURRING PREDICTIONS')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for imfile in tqdm.tqdm(test_img):
        batch = 0
        time = 0
        map_idx = 0
        gt_shape = Image.open(imfile).size[::-1]
        img = preprocess_images([imfile], inp_size[0], inp_size[1])
        preds = model.predict(img)
        if mode == 'multistream_concat':
            p = preds[time][batch][map_idx][:, :, 0]
        elif mode == 'simple':
            p = preds[0][batch][:,:,0]
        elif mode == 'singlestream':
            p = preds[0][batch][time][:,:,0]
        else:
            raise ValueError('Unknown mode')
        p = postprocess_predictions(p, gt_shape[0], gt_shape[1], blur, normalize=False, zero_to_255=True)
        p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
        p_img = p_norm*255
        hm_img = Image.fromarray(np.uint8(p_img), "L")
        print("heatmap shape", p_img.shape)

        imname = os.path.splitext(os.path.basename(imfile))[0] + "." + ext
        if test_img_base_path:
            relpath = os.path.dirname(imfile).replace(test_img_base_path, "")
            relpath = os.path.join(savedir, relpath)
            if not os.path.exists(relpath):
                os.makedirs(relpath)
            savepath = os.path.join(relpath, imname)
        else:
            savepath = os.path.join(savedir, imname)
        hm_img.save(savepath)

def predict_and_save_md(model, test_img, inp_size, savedir, mode='multistream_concat', blur=False, test_img_base_path="", times=[500, 3000, 5000], ext="png"):
    # if test_img_base_path is specified, then preserves the original
    # nested structure of the directory from which the stuff is pulled
    c=0
    if blur:
        print('BLURRING PREDICTIONS')
        if 'blur' not in savedir:
            savedir = savedir+'_blur'
    else:
        print('NOT BLURRING PREDICTIONS')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for imfile in tqdm.tqdm(test_img):

        batch = 0
        time = 0
        map_idx = 0
        gt_shape = Image.open(imfile).size[::-1]
        img = preprocess_images([imfile], inp_size[0], inp_size[1])
        preds = model.predict(img)
        for time, timelen in enumerate(times):
            if mode == 'multistream_concat':
                p = preds[time][batch][map_idx][:, :, 0]
            elif mode == 'simple':
                p = preds[0][batch][:,:,0]
            elif mode == 'singlestream':
                p = preds[0][batch][time][:,:,0]
            else:
                raise ValueError('Unknown mode')
            p = postprocess_predictions(p, gt_shape[0], gt_shape[1], blur, normalize=False)
            p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
            p_img = p_norm*255
            hm_img = Image.fromarray(np.uint8(p_img), "L")

            imname = os.path.splitext(os.path.basename(imfile))[0] + "_" + str(timelen) + "." + ext
            savepath = os.path.join(savedir, imname)
            hm_img.save(savepath)



def calculate_metrics(p, gt_map=None, gt_fix_map=None, gt_fix_points=None, gt_labels=None, p_labels=None):
    '''Calculates metrics for saliency given a SINGLE predicted map, its corresponding
    ground truth map (2D real valued np array), ground truth fixation map (2D binary np array),
    and ground truth fixation points (list of [x,y] coordinates corresponding to the fixation positions, 1 indexed)

    Inputs
    ------
    p: real valued 2D np array. Saliency map predicted by the model.
    gt_map: real valued 2D np array. Ground truth saliency map.
    gt_fix_map: binary 2D np array. Ground truth fixation map.
    gt_fix_points: list of [x,y] coords. 1 indexed.
    p_labels: 1D array, predicted one-hot label vector (softmax output)
    gt_labels: 1D array, true one-hot label vector

    Returns
    -------
    metrics: dictionary of metrics. The values of the metrics are encapsulated
        in a list element to play nicely with other functions in this file.
    '''
    metrics = {}

    p_norm = (p-np.min(p))/(np.max(p)-np.min(p))
    if np.max(gt_map)>1:
        gt_map = np.array(gt_map, dtype=np.float32)/255.

    if gt_map is not None:
        metrics['R2'] = [r2(gt_map, p_norm)]
        metrics['RMSE'] = [rmse(gt_map, p_norm)]
        metrics['CC'] = [cc_npy(gt_map, p_norm)]
        metrics['CC (saliconeval)'] = [cc_saliconeval(gt_map, p_norm)]
        metrics['KL'] = [kl_npy(gt_map, p_norm)]
        metrics['SIM'] = [sim_npy(gt_map, p_norm)]
    if gt_fix_map is not None:
        metrics['NSS'] = [nss_npy(gt_fix_map, p_norm)]
    if gt_fix_points is not None:
        metrics['NSS (saliconeval)'] = [nss_saliconeval(gt_fix_points, p_norm)]
        metrics['AUC'] = [auc_saliconeval(gt_fix_points, p)]
    if gt_labels is not None and p_labels is not None:
        metrics['Acc'] = [acc(gt_labels, p_labels)]
        metrics['Acc_per_class'] = [acc_per_class(gt_labels, p_labels)]


    return metrics

def get_stats_multiduration(model, 
                            gen_eval, 
                            mode='multistream_concat', 
                            blur=False,
                            start_at=False, 
                            compare_across_times=True, 
                            n_times=3,
                            return_top_bot_n=False, 
                            metrics_for_top_bot_n=['CC','KL'],
                            get_stats_per_dataset=False):
    '''Function to calculate metrics from a multiduration model. Calculates both the average over timesteps, as well
    as the per-timestep metrics. Can work in different modes:

    singlestream: assumes that both the model and the generator output a list of k elements (one per loss), where each element
    is a tensor of shape (bs, t, r, c, 1).

    multistream_concat: assumes that the model and generator both output a list of t elements, corresponding to
    t timesteps. Each one of those elements is a 5D tensor where the heatmap and fixation map (generator)
    or two copies of the pred map (model) are concatenated along the second dimension,
    resulting in a shape of (bs, 2, r, c, 1). To access the first fixmap from this 5D tensor, one would get
    the slice: (0,1,:,:,:). This mode should be used for 3stream models in concat mode, such as simple DCNNs
    with one output per timestep.
    '''
    # Setting starting point
    if start_at:
        gen = (next(gen_eval) for _ in range(start_at))
    else:
        gen = gen_eval

    # set up metric tracking code
    all_m = {}
    m_by_time = {t: {} for t in range(n_times)}
    m_per_dataset = {d: {} for d in CODECHARTS_SUB_DATASETS}
    m_per_dataset_by_time = {d: {t: {} for t in range(n_times)} for d in CODECHARTS_SUB_DATASETS}
    batch = 0
    top_n = {}
    bot_n = {}
    idx = 0

    # used for comparing cc across groups
    if compare_across_times:
        combos = {}
        for t_lower in range(n_times):
            for t_upper in range(t_lower +1, n_times):
                if t_lower not in combos:
                    combos[t_lower] = {}
                if t_upper not in combos[t_lower]:
                    combos[t_lower][t_upper] = []
    
    if isinstance(gen, Sequence):
        gen.return_names=True
        
    for dat in tqdm.tqdm_notebook(gen):
        if isinstance(gen, Sequence):
            imgs, outs, names_batch = dat
            
            # Gt maps need to recover their original size
            gt_map_batch = outs[0]
            gt_fix_map_batch = outs[-1]
            gt_fix_points_batch = None
            
        else:
            imgs, gt_map_batch, gt_fix_map_batch, gt_fix_points_batch, name = dat
            imgs = [imgs[0]]
            gt_map_batch = [gt_map_batch]
            gt_fix_map_batch = [gt_fix_map_batch]
            gt_fix_points_batch = [gt_fix_points_batch]
            names_batch = [name]
            
                        
        pred_batch = model.predict(imgs)[0] # the model outputs 4 maps (one for each loss), we take the first one as they're all the same here
                
        for batch_idx, prediction in enumerate(pred_batch):
            
            p_t_dict = {}
            m_t_dict = {}

            for t in range(n_times):
                if mode != 'singlestream':
                    raise ValueError('Mode %s not implemented' % mode)

                p_t = prediction[t]

                gt_map_t = np.squeeze(gt_map_batch[batch_idx][t])
                gt_fix_t = np.squeeze(gt_fix_map_batch[batch_idx][t])
                
                p_t = postprocess_predictions(p_t, gt_map_t.shape[0], gt_map_t.shape[1], blur, normalize=False)
                
                p_t_dict[t] = (p_t-np.min(p_t))/(np.max(p_t)-np.min(p_t))

                gt_fix_points_t = gt_fix_points_batch[batch_idx][t] if gt_fix_points_batch else None
                assert p_t.shape == gt_map_t.shape, "prediction and ground truth should have same dimensions, but got %s and %s" % (p_t.shape, gt_map_t.shape)

                m = calculate_metrics(p_t, gt_map_t, gt_fix_t, gt_fix_points_t)


                # If first pass, define metric dictionary as the dict returned from calculate_metrics
                for k,v in m.items():
                    all_m[k] = all_m.get(k, []) + v # append list to list or int to int
                    m_by_time[t][k] = m_by_time[t].get(k, []) + v # append list to list or int to int
                    if get_stats_per_dataset:
                        m_per_dataset[name_to_dataset(names_batch[batch_idx])][k] = m_per_dataset[name_to_dataset(names_batch[batch_idx])].get(k, []) + v 
                        m_per_dataset_by_time[name_to_dataset(names_batch[batch_idx])][t][k] = m_per_dataset_by_time[name_to_dataset(names_batch[batch_idx])][t].get(k, []) + v
                        
                m_t_dict[t] = m


            # Calculate CC Match
            all_m['CCM'] = all_m.get('CCM', []) + [cc_match_npy(gt_map_batch[batch_idx], prediction)]

            # calculate pairwise
            if compare_across_times:
                for t_lower, others in combos.items():
                    for t_higher in others.keys():
                        combos[t_lower][t_higher].append(cc_npy(p_t_dict[t_lower], p_t_dict[t_higher]))

    if return_top_bot_n:
        for me in metrics_for_top_bot_n:
            if me not in top_n:
                top_n[me] = []
            if len(top_n.get(me,[])) < return_top_bot_n:
                item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]] )
                top_n[me].append(item)
                cur_min_idx = np.argmin([it[0] for it in top_n[me]])
            elif np.mean(all_m[me][-n_times:]) > top_n[me][cur_min_idx][0]:
                del top_n[me][cur_min_idx]
                item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                top_n[me].append(item)
                cur_min_idx = np.argmin([it[0] for it in top_n[me]])
            if me not in bot_n:
                bot_n[me] = []
            if len(bot_n.get(me,[])) < return_top_bot_n:
                item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                bot_n[me].append(item)
                cur_max_idx = np.argmax([it[0] for it in bot_n[me]])
            elif np.mean(all_m[me][-n_times:]) < bot_n[me][cur_max_idx][0]:
                del bot_n[me][cur_max_idx]
                item = (np.mean(all_m[me][-n_times:]), idx, mt, imgs[0], hms_by_time, gt_map_batch, [combos[0][1][-1],combos[0][2][-1],combos[1][2][-1]])
                bot_n[me].append(item)
                cur_max_idx = np.argmax([it[0] for it in bot_n[me]])
            idx+=1



    print("Overall metrics:")
    for k,v in all_m.items():
        print("\t", k,':', np.mean(v))

    print()
    print("Metrics by time:")
    for t, m in m_by_time.items():
        print("\tTime %d" % t)
        for k, v in m.items():
            print("\t\t", k, ":", np.mean(v))

    print()
    if compare_across_times:
        print("CC across time groups:")
        for t_lower, others in combos.items():
            for t_higher, v in others.items():
                print("CC for times %d and %d" % (t_lower, t_higher), np.mean(v))
    
    if get_stats_per_dataset:
        print()
        print("Metrics per dataset:")
        for d in CODECHARTS_SUB_DATASETS:
            print("Dataset %s" % d)
            for k,v in m_per_dataset[d].items():
                print("\t", k,':', np.mean(v))
                
            for t, m in m_per_dataset_by_time[d].items():
                print("\tTime %d" % t)
                for k, v in m.items():
                    print("\t\t", k, ":", np.mean(v))

    if not return_top_bot_n and not get_stats_per_dataset:
        return all_m, m_by_time, combos
    
    if return_top_bot_n and get_stats_per_dataset:
        return all_m, m_by_time, combos, m_per_dataset, m_per_dataset_by_time, top_n, bot_n
    elif get_stats_per_dataset:
        return all_m, m_by_time, combos, m_per_dataset, m_per_dataset_by_time
    elif return_top_bot_n:
        return all_m, m_by_time, combos, top_n, bot_n
    else:
        return all_m, m_by_time, combos
        

def get_stats_oneduration(model, 
                          gen_eval, 
                          mode='multistream_concat',
                          blur=False, 
                          start_at=False, 
                          t=0, 
                          uses_fixs=True, 
                          uses_labels = False, 
                          eval_imgs=None, 
                          plot_while_eval=False):
    ''' Function to calculate metrics from a model, based on the model object and a generator.
    This function will always assume that the model is trained on a dataset with only one timestep. If the
    model and generator given output multiple timesteps, the function will assume that all maps are the same
    for each timestep and will take the t-eth one.

    The function can operate in different modes, depending on the shape and form of the output
    of the generator and model:

    multistream_concat: assumes that the model and generator both output a list of t elements, corresponding to
    t timesteps. Each one of those elements is a 5D tensor where the heatmap and fixation map (generator)
    or two copies of the pred map (model) are concatenated along the second dimension,
    resulting in a shape of (bs, 2, r, c, 1). To access the first fixmap from this 5D tensor, one would get
    the slice: (0,1,:,:,:). This mode should be used for 3stream models in concat mode, such as simple DCNNs
    with one output per timestep.

    singlestream: assumes that the model and generator output a list of k 5D tensors. Each
    tensor matches one loss, and their shapes are (bs, time, r, c, 1). As this function considers that all t maps are
    the same, it will slice this tensor at (bs, ~t~, r, c, 1).

    single: For when the model doesn't deal with timesteps. The model should output a list of k
    elements, each corresponding ot a loss of the model. The generator should also output k ground truths,
    one for each loss.

    Inputs
    ------
    model: a Keras model.
    gen_eval: a geneartor with data to evaluate. Can be a Keras generator outputing imgs, maps and fixmaps or
    a python generator outputting imgs, maps, fixmaps and fixlocs (x,y).
    mode: str. mode to use.
    blur: bool. whether to blur the predictions before evaluating.
    start_at: int. Idx to start in the generator.
    t: int. tensor idx to get from the time dimension, if existing.

    Returns
    -------
    metrics: a dictionary of metrics, where for each metric, a list of values for each element in the set is available.
    '''

    c = 0
    first_pass = True

    # Setting starting point
    if start_at:
        gen = (next(gen_eval) for _ in range(start_at))
    else:
        gen = gen_eval

    # Iterating over generator
    metrics = {}
    for dat in tqdm.tqdm_notebook(gen):

        ## Get ground truths
        if isinstance(gen, Sequence):   # If the generator is a Keras Sequence
            gt_fix_points_batch = None
            imgs, gt_set = dat
            if mode == 'multistream_concat':
                gt_map_batch = gt_set[t][:,0,...] #batch of gt_maps
                if uses_fixs:
                    gt_fix_map_batch = gt_set[t][:,-1,...] #batch of gt_fix_maps
            if mode == 'singlestream':
                gt_map_batch = gt_set[0][:,t,...] #batch of gt_maps
                if uses_fixs:
                    gt_fix_map_batch = gt_set[-1][:,t,...] #batch of gt_fix_maps
            elif mode == 'simple':
                gt_map_batch = gt_set[0]
                if uses_fixs:
                    gt_fix_map_batch = gt_set[-1] if not uses_labels else gt_set[-2]
                if uses_labels:
                    gt_labels_batch = gt_set[-1]
            else:
                raise ValueError('Unknown mode: '+str(mode))
        else:  # If the generator is a conventional python generator
            imgs, gt_map_batch, gt_fix_map_batch, gt_fix_points_batch = dat

        ## Get prediction batch
        prediction = model.predict(imgs)
        if mode == 'multistream_concat':
            pred_batch = prediction[t][:,0,...]
        elif mode == 'singlestream':
            pred_batch = prediction[0][:,t,...]
        elif mode == 'simple':
            pred_batch = prediction[0]
            if uses_labels:
                p_labels_batch = prediction[-1]
            if not isinstance(gen, Sequence):
                pred_batch = pred_batch[:,:,:,0]
        else:
            raise ValueError('Unknown mode: '+str(mode))

        for i in range(len(pred_batch)): # loop over batch
            # the 0 is to get rid of the copy
            p = pred_batch[i]
            gt_map = gt_map_batch[i]
            if uses_fixs:
                gt_fix = gt_fix_map_batch[i]
                gt_fix = gt_fix.squeeze()
            else:
                gt_fix = None
            if uses_labels:
                p_labels = p_labels_batch[i]
                gt_labels = gt_labels_batch[i]
            else:
                p_labels = None
                gt_labels = None
            p = p.squeeze()
            gt_map = gt_map.squeeze()

            gt_size = gt_map.shape
            p = postprocess_predictions(p, gt_size[0], gt_size[1], blur, normalize=False)

            ## # DEBUG:

            if plot_while_eval:
                plt.figure(figsize=[15,8])
                plt.subplot(1,2,1)
                plt.imshow(gt_map)
                plt.subplot(1,2,2)
                plt.imshow(p)
                plt.show()

            if gt_fix_points_batch:
                gt_fix_points = gt_fix_points_batch[i]
            else:
                gt_fix_points = None


            assert p.shape == gt_map.shape, "prediction and ground truth should have same dimensions, but are %s and %s" % (p.shape, gt_map.shape)

            m = calculate_metrics(p, gt_map=gt_map, gt_fix_map=gt_fix, gt_fix_points=gt_fix_points, gt_labels=gt_labels, p_labels=p_labels)

            # If first pass, define metric dictionary as the dict returned from calculate_metrics
            for k,v in m.items():
                metrics[k] = metrics.get(k, []) + v # append list to list or int to int

    # Print results
    for k,v in metrics.items():
        if k == 'Acc_per_class':
            v = np.array(v)
            accs=np.zeros(v.shape[1])
            for i in range(v.shape[1]):
                accs[i] = np.sum(v[:,i]==True)/np.sum(v[:,i]!=None)
            print(k,':',accs)
        else:
            print(k,':', np.mean(v))

    return metrics



#### LOGGING FUNCTION ####

def log_metrics(metrics, dataset, n_elems, model_name, ckpt, txt_path='../metric_logs.txt',
                lossnames=['KL','CC','NSS'], print_=True, eval_with='eval_gen', blur=False, simple_log = True):
    import datetime
    
    m = metrics

    with open(txt_path, 'a+') as f:
        f.write('\n\nMODEL: %s' % model_name)
        if print_: print('\nMODEL:',model_name)
        f.write('\nCkpt: %s' % ckpt)
        f.write('\nEvaluated on %d elems of %s with fct %s and blur=%s on: %s' % (n_elems, dataset, eval_with, blur, str(datetime.datetime.now())))
        if print_: print('Ckpt:', ckpt)

        if simple_log:
            for k,v in m.items():
                if k == 'Acc_per_class':
                    v = np.array(v)
                    accs=np.zeros(v.shape[1])
                    for i in range(v.shape[1]):
                        accs[i] = np.sum(v[:,i]==True)/np.sum(v[:,i]!=None)
                    f.write('\n%s: %s' % (k, accs))
                    if print_: print(k,':',accs)
                else:
                    f.write('\n%s: %.4f' % (k, np.mean(v)))
                    if print_: print('%s: %.4f' % (k, np.mean(v)))
        else: 
            all_m = m[0]
            m_by_time = m[1]
            if len(m)>=3:
                compare_across_times = True
                combos = m[2]
                
                if len(m) > 3:
                    get_stats_per_dataset = True
                    m_per_dataset = m[3]
                    m_per_dataset_by_time = m[4]
            else:
                compare_across_times = False
            
            f.write("\n\nOverall metrics:")
            for k,v in all_m.items():
                f.write("\n\t %s: %.4f" %(k, np.mean(v)))

            f.write("\n\nMetrics by time:")
            for t, met in m_by_time.items():
                f.write("\n\tTime %d" % t)
                for k, v in met.items():
                    f.write("\n\t\t %s: %.4f" %(k, np.mean(v)))

            if compare_across_times:
                f.write("\n\nCC across time groups:")
                for t_lower, others in combos.items():
                    for t_higher, v in others.items():
                        f.write("\nCC for times %d and %d: %.4f" % (t_lower, t_higher, np.mean(v)))
                    
            if get_stats_per_dataset:
                f.write("\nMetrics per dataset:")
                for d in CODECHARTS_SUB_DATASETS:
                    f.write("\nDataset %s" % d)
                    for k,v in m_per_dataset[d].items():
                        f.write("\n\t %s: %.4f" %(k, np.mean(v)))

                    for t, m in m_per_dataset_by_time[d].items():
                        f.write("\n\tTime %d" % t)
                        for k, v in m.items():
                            f.write("\n\t\t %s: %.4f" %(k, np.mean(v)))                   

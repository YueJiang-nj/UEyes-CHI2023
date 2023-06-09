import os
import numpy as np
import json

def load_datasets_singleduration(dataset, bp='/netpool/work/gpu-2/users/wangyo/datasets', return_test=False):
    fix_as_mat=False
    fix_key=None

    if dataset == 'gdi':
        print('Using GDI')

        uses_fix =False
        has_classes = False

        img_path_train = os.path.join(bp,'GDI/gd_train')
        imp_path_train = os.path.join(bp,'GDI/gd_imp_train')
        img_path_val = os.path.join(bp,'GDI/gd_val')
        imp_path_val = os.path.join(bp,'GDI/gd_imp_val')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])
        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        # Dummy variables
        fix_filenames_train = None #np.array([])
        fix_filenames_val = None #np.array([])


    elif dataset == 'graphs':
        print('Using GRAPHS')

        uses_fix=False
        has_classes = False

        img_path_train = '../../predimportance_shared/datasets/graphs/graphs_train'
        imp_path_train = '../../predimportance_shared/datasets/graphs/graphs_imp_train'
        img_path_val = '../../predimportance_shared/datasets/graphs/graphs_val'
        imp_path_val = '../../predimportance_shared/datasets/graphs/graphs_imp_val'

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])
        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        # Dummy variables
        fix_filenames_train = None #np.array([])
        fix_filenames_val = None # np.array([])

    elif dataset == 'salicon':
        print('Using SALICON')

        uses_fix=True
        has_classes = False

        img_path_train = os.path.join(bp, 'Salicon', 'images', 'train')
        imp_path_train = os.path.join(bp, 'Salicon', 'maps', 'train')

        img_path_val = os.path.join(bp, 'Salicon', 'images', "val")
        imp_path_val = os.path.join(bp, 'Salicon', 'maps', 'val')

        img_path_test = os.path.join(bp, 'Salicon','images', 'test')

        fix_path_train = os.path.join(bp, 'Salicon', 'fixations', 'train')
        fix_path_val = os.path.join(bp, 'Salicon', 'fixations', 'val')

        fixcoords_path_train = os.path.join(bp, 'Salicon', 'fixations', 'train')
        fixcoords_path_val = os.path.join(bp, 'Salicon', 'fixations', 'val')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])

        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        img_filenames_test = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])

        fix_filenames_train = sorted([os.path.join(fix_path_train, f) for f in os.listdir(fix_path_train) if f.endswith('.png')])
        fix_filenames_val = sorted([os.path.join(fix_path_val, f) for f in os.listdir(fix_path_val) if f.endswith('.png')])

        fixcoords_filenames_train = sorted([os.path.join(fixcoords_path_train, f) for f in os.listdir(fixcoords_path_train)])
        fixcoords_filenames_val = sorted([os.path.join(fixcoords_path_val, f) for f in os.listdir(fixcoords_path_val)])


        print('Length of loaded files:')
        print('train images:', len(img_filenames_train))
        print('train maps:', len(imp_filenames_train))
        print('val images:', len(img_filenames_val))
        print('val maps:', len(imp_filenames_val))
        print('test images', len(img_filenames_test))

        print('train fixs:', len(fix_filenames_train))
        print('val fixs:', len(fix_filenames_val))
        print('train fixcoords:', len(fixcoords_filenames_train))
        print('val fixcoords:', len(fixcoords_filenames_val))

    elif dataset == 'UMSI_SALICON':
        print('Using SALICON')

        uses_fix=False
        has_classes = True

        img_path_train = os.path.join(bp, 'Salicon', 'images', 'train')
        imp_path_train = os.path.join(bp, 'Salicon', 'maps', 'train')

        img_path_val = os.path.join(bp, 'Salicon', 'images', "val")
        imp_path_val = os.path.join(bp, 'Salicon', 'maps', 'val')

        img_path_test = os.path.join(bp, 'Salicon','images', 'test')

        img_filenames_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        imp_filenames_train = sorted([os.path.join(imp_path_train, f) for f in os.listdir(imp_path_train)])

        img_filenames_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        imp_filenames_val = sorted([os.path.join(imp_path_val, f) for f in os.listdir(imp_path_val)])

        img_filenames_test = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])

        fix_filenames_train = None #np.array(['dummy']*len(img_filenames_train))
        fix_filenames_val = None #np.array(['dummy']*len(img_filenames_val))


        print('Length of loaded files:')
        print('train images:', len(img_filenames_train))
        print('train maps:', len(imp_filenames_train))
        print('val images:', len(img_filenames_val))
        print('val maps:', len(imp_filenames_val))
        print('test images', len(img_filenames_test))



    elif dataset == 'mit1003':
        print('Using MIT1003')
        uses_fix=True
        has_classes = False

        img_path = os.path.join(bp, "mit1003/ALLSTIMULI")
        imp_path = os.path.join(bp, 'mit1003/ALLFIXATIONMAPS')
        fix_path = os.path.join(bp, 'datasets/mit1003/ALLFIXATIONMAPS')

        imgs = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpeg')])
        maps = sorted([os.path.join(imp_path, f) for f in os.listdir(imp_path) if 'fixMap' in f and f.endswith('.jpg')])
        fixs = sorted([os.path.join(imp_path, f) for f in os.listdir(imp_path) if 'fixPts' in f and f.endswith('.jpg')])

        # Randomly shuffling mit1003
        np.random.seed(42)
        idxs = list(range(len(imgs)))
        np.random.shuffle(idxs)
        imgs = np.array(imgs)[idxs]
        maps = np.array(maps)[idxs]
        fixs = np.array(fixs)[idxs]

        img_filenames_train = imgs[:903]
        imp_filenames_train = maps[:903]
        fix_filenames_train = fixs[:903]
        img_filenames_val = imgs[903:]
        imp_filenames_val = maps[903:]
        fix_filenames_val = fixs[903:]

        #print('Length of loaded files:')
        #print('train images:', len(img_filenames_train))
        #print('train maps:', len(imp_filenames_train))
        #print('val images:', len(img_filenames_val))
        #print('val maps:', len(imp_filenames_val))
        ##print('test images', len(img_filenames_test))

        #print('train fixs:', len(fix_filenames_train))
        #print('val fixs:', len(fix_filenames_val))
        #print('train fixcoords:', len(fix_filenames_train))
        #print('val fixcoords:', len(fix_filenames_val))


    elif dataset == 'mit300':
        img_path = os.path.join(bp, 'mit300')
        #img_path = '../../predimportance_shared/datasets/mit300'
        imgs = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')])
        print("num images", len(imgs))
        return imgs, None, None, None, None, None, False, False, False


    elif dataset == 'imp1k':
        print('Using imp1k')

        k = 40
        uses_fix=False
        has_classes=True

        img_path = os.path.join(bp, 'imp1k', 'imgs')
        imp_path = os.path.join(bp, 'imp1k', 'maps')

        img_filenames_train = np.array([])
        imp_filenames_train = np.array([])
        img_filenames_val = np.array([])
        imp_filenames_val = np.array([])

        use_tts_file=True

        if not use_tts_file:
            for f in os.listdir(img_path):
                print('Categ:',f)
                imgs = sorted([os.path.join(img_path, f, i) for i in os.listdir(os.path.join(img_path,f)) if i.endswith(('.png','.jpg'))])
                maps = sorted([os.path.join(imp_path, f, i) for i in os.listdir(os.path.join(imp_path,f)) if i.endswith(('.png','.jpg'))])

                idxs = list(range(len(imgs)))
                np.random.shuffle(idxs)
                imgs = np.array(imgs)[idxs]
                maps = np.array(maps)[idxs]

                img_filenames_train = np.concatenate([img_filenames_train,imgs[:-k]], axis=None)
                img_filenames_val = np.concatenate([img_filenames_val,imgs[-k:]], axis=None)
                imp_filenames_train = np.concatenate([imp_filenames_train,maps[:-k]], axis=None)
                imp_filenames_val = np.concatenate([imp_filenames_val,maps[-k:]], axis=None)

        else:
            with open(os.path.join(bp, 'imp1k', 'train_test_split_imp1k.json'), 'r') as f:
                tt_s = json.load(f)

            train_names = tt_s[0]
            test_names = tt_s[1]

            img_filenames_train = [os.path.join(img_path, n) for n in train_names]
            imp_filenames_train = [os.path.join(imp_path, n) for n in train_names]

            img_filenames_val = [os.path.join(img_path, n) for n in test_names]
            imp_filenames_val = [os.path.join(imp_path, n) for n in test_names]


        # Dummy variables
        fix_filenames_train = None #np.array(['dummy']*len(img_filenames_train))
        fix_filenames_val = None #np.array(['dummy']*len(img_filenames_val))

    elif dataset == 'cat2000':
        print('Using CAT2000')
        uses_fix=True
        fix_as_mat=True
        fix_key="fixLocs"
        has_classes = True

        # TODO: MAKE SURE THAT THE VAL SET IS ALWAYS THE SAME
        np.random.seed(42)
        img_path = os.path.join(bp, 'cat2000', 'Stimuli')
        imp_path = os.path.join(bp, 'cat2000', 'FIXATIONMAPS')
        fix_path = os.path.join(bp, 'cat2000', 'FIXATIONLOCS')
        img_path_test = os.path.join(bp, 'cat2000', 'testStimuli')

        img_filenames_train = np.array([])
        imp_filenames_train = np.array([])
        fix_filenames_train = np.array([])
        img_filenames_val = np.array([])
        imp_filenames_val = np.array([])
        fix_filenames_val = np.array([])
        img_filenames_test = np.array([])

        for f in os.listdir(img_path):
            #print('Categ:',f)
            imgs = sorted([os.path.join(img_path, f, i) for i in os.listdir(os.path.join(img_path,f)) if i.endswith('.jpg')])
            maps = sorted([os.path.join(imp_path, f, i) for i in os.listdir(os.path.join(imp_path,f)) if i.endswith('.jpg')])
            fixs = sorted([os.path.join(fix_path, f, i) for i in os.listdir(os.path.join(fix_path,f)) if i.endswith('.mat')])

            idxs = list(range(len(imgs)))
            np.random.shuffle(idxs)
            imgs = np.array(imgs)[idxs]
            maps = np.array(maps)[idxs]
            fixs = np.array(fixs)[idxs]

            img_filenames_train = np.concatenate([img_filenames_train,imgs[:-10]], axis=None)
            img_filenames_val = np.concatenate([img_filenames_val,imgs[-10:]], axis=None)
            imp_filenames_train = np.concatenate([imp_filenames_train,maps[:-10]], axis=None)
            imp_filenames_val = np.concatenate([imp_filenames_val,maps[-10:]], axis=None)
            fix_filenames_train = np.concatenate([fix_filenames_train,fixs[:-10]], axis=None)
            fix_filenames_val = np.concatenate([fix_filenames_val,fixs[-10:]], axis=None)

        for f in os.listdir(img_path_test):
            new_files = sorted([os.path.join(img_path_test, f, i) for i in os.listdir(os.path.join(img_path_test, f)) if i.endswith('.jpg')])
            img_filenames_test = np.concatenate([img_filenames_test, new_files], axis=None)

    print('Length of loaded files:')
    print('train images:', len(img_filenames_train))
    print('train maps:', len(imp_filenames_train))
    print('val images:', len(img_filenames_val))
    print('val maps:', len(imp_filenames_val))
    if return_test:
        print('test images', len(img_filenames_test))
    
    if fix_filenames_train and fix_filenames_val:
        print('train fixs:', len(fix_filenames_train))
        print('val fixs:', len(fix_filenames_val))
#        print('train fixcoords:', len(fixcoords_filenames_train))
#        print('val fixcoords:', len(fixcoords_filenames_val))


    if return_test:
        return img_filenames_train, imp_filenames_train, fix_filenames_train, img_filenames_val, imp_filenames_val, fix_filenames_val, img_filenames_test, uses_fix, fix_as_mat, fix_key, has_classes
    else:
        return img_filenames_train, imp_filenames_train, fix_filenames_train, img_filenames_val, imp_filenames_val, fix_filenames_val, uses_fix, fix_as_mat, fix_key, has_classes


def load_multiduration_data(img_folder, map_folder, fix_folder, times=[500,3000,5000]):
    '''
    Takes in a saliency heatmap groundtruth folder and a fixation groundtruth folder
    and searches for the specified exposure durations in `times`. The structure
    of the ground truth folders should be:
    map_or_fix_folder
        |_ time1
            |_ map1.png
            |_ ...
        |_ time2
        |_ time3
        |_ ...

    Inputs
    ------
    img_folder: string. Path to the folder containing input images. The names of
        these images should maimg_file://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-formattch the names of the corresponding fixations and heatmaps.
    map_folder: string. Path to the ground truth map folder. That folder should
        contain subfolders named as times and containing the images.
    fix_folder: string. Path to the ground truth fix folder. Idem as above.
    times: array of numbers or strings indicating what times to use. Should match
        the names of subfolders in map and fix folders.

    Returns:
    --------
    img_filenames: array of strings containing paths to images.
    map_filenames: array of length `times` containing arrays of strings corresponding to the paths to each image.
    fix_filenames: idem as above but for fixations.
    '''
    print('img_folder', img_folder)
    img_filenames = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.npy')])
    n_expected = len(img_filenames)

    map_filenames = []
    fix_filenames = []

    avlb_times = sorted([int(elt) for elt in os.listdir(map_folder)])
    print('avlb_times',avlb_times)

    for t in avlb_times:
        if t in times:
            print('APPENDING IMAGES FOR TIME:', t)
            print('DATA_SCALE:', n_expected)
            t_map_filenames = sorted([os.path.join(map_folder, str(t), f) for f in os.listdir(os.path.join(map_folder, str(t))) if f.endswith('.png') or f.endswith('.npy')])
            t_fix_filenames = sorted([os.path.join(fix_folder, str(t), f) for f in os.listdir(os.path.join(fix_folder, str(t))) if f.endswith('.png') or f.endswith('.npy')])
            assert len(t_map_filenames) == n_expected
            assert len(t_fix_filenames) == n_expected

            map_filenames.append(t_map_filenames)
            fix_filenames.append(t_fix_filenames)

    # check that we have even numbers of everything
    return img_filenames, map_filenames, fix_filenames


def load_datasets_multiduration(dataset, times, bp='/netpool/work/gpu-2/users/wangyo/datasets', test_splits=[0]):
    fix_as_mat = False
    fix_key = ""
    uses_fix = True
    use_accum = False
    accum_suffix = "_accum" if use_accum else ""

    if dataset == 'salicon':
        print('Using SALICON')
        rep = len(times)

        img_path_train = '../../predimportance_shared/datasets/salicon/train'
        map_path_train = '../../predimportance_shared/datasets/salicon/train_maps'
        img_path_val = '../../predimportance_shared/datasets/salicon/val'
        map_path_val = '../../predimportance_shared/datasets/salicon/val_maps'

        fix_path_train = '../../predimportance_shared/datasets/salicon/train_fix_png'
        fix_path_val = '../../predimportance_shared/datasets/salicon/val_fix_png'

        img_files_train = sorted([os.path.join(img_path_train, f) for f in os.listdir(img_path_train)])
        map_files_train = [sorted([os.path.join(map_path_train, f) for f in os.listdir(map_path_train)])]*rep
        fix_files_train = [sorted([os.path.join(fix_path_train, f) for f in os.listdir(fix_path_train) if f.endswith('.png')])]*rep
        img_files_val = sorted([os.path.join(img_path_val, f) for f in os.listdir(img_path_val)])
        map_files_val = [sorted([os.path.join(map_path_val, f) for f in os.listdir(map_path_val)])]*rep
        fix_files_val = [sorted([os.path.join(fix_path_val, f) for f in os.listdir(fix_path_val) if f.endswith('.png')])]*rep
        img_files_test = []
        map_files_test = []
        fix_files_test = []

    elif dataset == 'codecharts':
        bp = os.path.join(bp, 'codecharts_data')

        use_accum = False
        accum_suffix = "_accum" if use_accum else ""

        n_val=50

        img_path = os.path.join(bp, "raw_img" + accum_suffix)
        map_path = os.path.join(bp, "heatmaps" + accum_suffix)
        fix_path = os.path.join(bp, "fix_maps" + accum_suffix)

        img_files, map_files, fix_files = load_multiduration_data(img_path, map_path, fix_path, times=times)
        #test_data = load_multiduration_data(img_path_val, map_path_val, fix_path_val, times=[500,3000,5000])

        # data divided in 4 splits of 250 images each. when test_splits=[0], we use sets [1,2,3] for training and [0] for test. 
        with open("%s/splits.json" % bp) as infile:
            splits = json.load(infile)
            non_test_splits = [s for s in splits.keys() if int(s) not in test_splits]

            train = [elt for i in non_test_splits for elt in splits[str(i)]]
            test = set([elt for i in test_splits for elt in splits[str(i)]])
            val = set(train[:n_val])
            train = set(train[n_val:])

        def _imname(imfile):
            return os.path.splitext(os.path.basename(imfile))[0]

        def img_in(imfile, imset):
            return _imname(imfile) in (_imname(elt) for elt in imset)

        img_files_train = [f for f in img_files if img_in(f, train)]
        img_files_val = [f for f in img_files if img_in(f, val)]
        img_files_test = [f for f in img_files if img_in(f, test)]

        map_files_train = []
        map_files_val = []
        map_files_test = []
        fix_files_train = []
        fix_files_val = []
        fix_files_test = []

        for t in range(len(map_files)):
            for f in map_files[t]:
                bn = os.path.basename(f)
            map_files_train.append([f for f in map_files[t] if img_in(f, train)])
            map_files_val.append([f for f in map_files[t] if img_in(f, val)])
            map_files_test.append([f for f in map_files[t] if img_in(f, test)])

            fix_files_train.append([f for f in fix_files[t] if img_in(f, train)])
            fix_files_val.append([f for f in fix_files[t] if img_in(f, val)])
            fix_files_test.append([f for f in fix_files[t] if img_in(f, test)])

    elif dataset == "salicon_md" or dataset == "salicon_md_fixations":
        bp_md = os.path.join(bp, 'salicon_md')

        dsets = {}
        for mode in ["train", "val"]:
            img_path = os.path.join(bp_md, mode, "raw_img" + accum_suffix)
            map_path = os.path.join(bp_md, mode, "heatmaps" + accum_suffix)
            fix_path = os.path.join(bp_md, mode, "fix_maps" + accum_suffix)
            data = load_multiduration_data(img_path, map_path, fix_path, times=times)
            dsets[mode] = data

        img_files_train = dsets["train"][0]
        map_files_train = dsets["train"][1]
        fix_files_train = dsets["train"][2]

        img_files_val = dsets["val"][0]
        map_files_val = dsets["val"][1]
        fix_files_val = dsets["val"][2]

        # load the test images (only have images)
        img_path_test = os.path.join(bp, 'salicon', 'images', 'test')
        img_files_test = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])
        map_files_test = None
        fix_files_test = None

    elif "massvis" in dataset:
        bp_md = os.path.join(bp, dataset)

        dsets = {}
        for mode in ["train", "val"]:
            img_path = os.path.join(bp_md, mode, "raw_img" + accum_suffix)
            map_path = os.path.join(bp_md, mode, "heatmaps" + accum_suffix)
            fix_path = os.path.join(bp_md, mode, "fix_maps" + accum_suffix)
            data = load_multiduration_data(img_path, map_path, fix_path, times=times)
            dsets[mode] = data

        img_files_train = dsets["train"][0]
        map_files_train = dsets["train"][1]
        fix_files_train = dsets["train"][2]

        img_files_val = dsets["val"][0]
        map_files_val = dsets["val"][1]
        fix_files_val = dsets["val"][2]


        # load the test images (only have images)
        img_path_test = os.path.join(bp_md, 'test')

        img_files_test = sorted([os.path.join(img_path_test, f) for f in os.listdir(img_path_test)])
        map_files_test = None
        fix_files_test = None
    else:
        raise ValueError("Unknown dataset. Possible values are codecharts, salicon or salicon_md")

    print("img_files_train: %d" % len(img_files_train))
    print("img_files_val: %d" % len(img_files_val))
    print("img_files_test: %d" % len(img_files_test))

    print("map_files_train: ", [len(elt) for elt in map_files_train])
    print("map_files_val: ", [len(elt) for elt in map_files_val])
    print("map_files_test: ", [len(elt) for elt in map_files_test] if  map_files_test else map_files_test)

    print("fix_files_train: ", [len(elt) for elt in fix_files_train])
    print("fix_files_val: ", [len(elt) for elt in fix_files_val])
    print("fix_files_test: ", [len(elt) for elt in fix_files_test] if fix_files_test else fix_files_test)

    return {
        'img_files_train': img_files_train,
        'map_files_train': map_files_train,
        'fix_files_train': fix_files_train,
        'img_files_val': img_files_val,
        'map_files_val': map_files_val,
        'fix_files_val': fix_files_val,
        'img_files_test': img_files_test,
        'map_files_test': map_files_test,
        'fix_files_test': fix_files_test,
        'uses_fix': uses_fix,
        'fix_as_mat': fix_as_mat,
        'fix_key': fix_key
    }

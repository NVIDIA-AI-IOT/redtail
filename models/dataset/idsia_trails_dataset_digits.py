# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import argparse
import os
import random
import sys

train_datasets = ['001', '002', '004', '005', '006', '007', '009']
#train_datasets = ['001', '002'] # For testing

val_datasets = ['003', '008', '010']
#val_datasets = ['010'] # For testing

test_datasets = ['012']

labels = {val: idx for (idx, val) in enumerate(['lc', 'sc', 'rc'])}

#root_dir = '/data/redtail/datasets/TrailDatasetIDSIA_GOLD/'

def enumerate_images(path, remove_prefix=''):
    """
    Enumerates images recursively given the path.
    """ 
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                prefix = root[len(remove_prefix):]
                yield os.path.join(prefix, file)

def list_dir(root_dir, dir_path, label):
    """
    Returns sorted list of files for a particular label.
    Sort files numerically rather than lexicographically to enable
    partitioning later. Assuming that source files are named more or less
    in order of frames from the trail.
    """
    path = os.path.join(dir_path, os.path.join('videos', label))
    return sorted(list(enumerate_images(path, root_dir)), 
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].replace('frame', '')))

def sample_balance_dir(root_dir, path, sample_interval=1):
    """
    Returns a balanced, undersampled list of files from a directory specified by path.
    """
    res = {}
    # Get files for each label.
    for l in labels.iterkeys():
        res[l] = list_dir(root_dir, path, l)
    # Balance class entries for the current dir
    # REVIEW alexeyk: this cuts off head/tail of larger sets, is this right?
    min_size = min([len(res[l]) for l in res])
    for l in labels.iterkeys():
        cur_size = len(res[l])
        if cur_size > min_size or sample_interval > 1:
            start = (cur_size - min_size) / 2
            res[l] = res[l][start:(start + min_size):sample_interval]
    return res

def sample_dir(root_dir, path, sample_interval=1):
    """
    Returns a sampled list of files from a directory specified by path.
    """
    res = {}
    # Get files for each label.
    for l in labels.iterkeys():
        res[l] = list_dir(root_dir, path, l)
    if sample_interval > 1:
        for l in labels.iterkeys():
            res[l] = res[l][::sample_interval]
    return res

def write_map_file_with_undersampling(map_path, root_dir, directories, max_num_items=10000, sample_interval=1):
    """
    Creates map file out of files in directories with balancing and undersampling.
    """
    dir_files = {}
    # Get clean list of files from all directories.
    for d in directories:
        dir_path = os.path.join(root_dir, d)
        print 'Processing ' + dir_path
        dir_files[d] = sample_balance_dir(root_dir, dir_path, sample_interval)
    # Balance directories. Each directory has equal number of files for each
    # label so just take count from first label.
    min_size = min([len(v[labels.keys()[0]]) for v in dir_files.itervalues()]) 
    max_per_dir_per_class = min(max_num_items / (len(dir_files) * len(labels)), min_size)
    print('Using {} iterms per directory per class.'.format(max_per_dir_per_class))

    with open(map_path, 'w') as f:
        for dir in dir_files.itervalues():
            for lab_dir in dir.iteritems():
                for path in lab_dir[1][:max_per_dir_per_class]:
                    f.write('{} {}\n'.format(path, labels[lab_dir[0]]))

def write_map_file_with_oversampling(map_path, root_dir, directories, max_num_items=100000, sample_interval=1):
    """
    Creates map file out of files in directories with balancing and oversampling.
    """
    dir_files = {}
    # Get clean list of files from all directories.
    for d in directories:
        dir_path = os.path.join(root_dir, d)
        print 'Processing ' + dir_path
        dir_files[d] = sample_dir(root_dir, dir_path, sample_interval)
    # Balance directories.
    # Find the largest directory size.
    max_size = max([len(d) for parent_dir in dir_files.itervalues() for d in parent_dir.itervalues()])
    max_per_dir_per_class = min(max_num_items / (len(dir_files) * len(labels)), max_size)
    print('Using {} iterms per directory per class.'.format(max_per_dir_per_class))

    with open(map_path, 'w') as f:
        for dir in dir_files.itervalues():
            for lab_dir in dir.iteritems():
                cur_size = len(lab_dir[1])
                if cur_size >= max_per_dir_per_class:
                    for path in lab_dir[1][:max_per_dir_per_class]:
                        f.write('{} {}\n'.format(path, labels[lab_dir[0]]))
                else:
                    numIter = (max_per_dir_per_class + cur_size - 1) / cur_size
                    i = 0
                    while i < max_per_dir_per_class:
                        path = lab_dir[1][i % cur_size]
                        f.write('{} {}\n'.format(path, labels[lab_dir[0]]))
                        i += 1

def write_full_dir_map_file(map_path, root_dir, directories, max_num_items=100000, sample_interval=1):
    """
    Creates map file out of files in directories.
    """
    dir_files = {}
    # Get clean list of files from all directories.
    for d in directories:
        dir_path = os.path.join(root_dir, d)
        print 'Processing ' + dir_path
        dir_files[d] = sample_dir(root_dir, dir_path, sample_interval)

    cur_items = 0
    with open(map_path, 'w') as f:
        for d in dir_files.itervalues():
            for lab, idx in labels.iteritems():
                for path in d[lab]:
                    f.write('{} {}\n'.format(path, idx))
                    cur_items += 1
                    if cur_items > max_num_items:
                        return

def write_000_map_file(root_dir, map_path):
    with open(map_path, 'w') as f:
        for lab, idx in labels.iteritems():
            files = enumerate_images(os.path.join(root_dir, os.path.join('000/videos', lab)), root_dir)
            for path in files:
                f.write('{} {}\n'.format(path, idx))


def main(sample_type, root_dir, train_map, max_train_items, val_map, max_val_items, sample_interval):
    print('Creating train map...')
    if sample_type == 'undersample':
        write_map_file_with_undersampling(train_map, root_dir, train_datasets,
                                          max_num_items=max_train_items, sample_interval=sample_interval)
    elif sample_type == 'oversample':
        write_map_file_with_oversampling(train_map, root_dir, train_datasets,
                                         max_num_items=max_train_items, sample_interval=sample_interval)
    else:
        assert False, sample_type
    print('Creating validation map...')
    # Don't do under/oversampling for validation dataset.
    write_full_dir_map_file(val_map, root_dir, val_datasets,
                            max_num_items=max_val_items, sample_interval=sample_interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create map files from IDSIA Trails dataset.')
    parser.add_argument('src_root_dir')
    parser.add_argument('path_to_train_map')
    parser.add_argument('max_train_items', type=int)
    parser.add_argument('path_to_val_map')
    parser.add_argument('max_val_items', type=int)
    parser.add_argument('-s', '--sample-type', choices=['undersample', 'oversample'], default='undersample')
    parser.add_argument('-i', '--sample-interval', type=int, default=1)
    args = parser.parse_args()

    main(args.sample_type, args.src_root_dir, args.path_to_train_map, args.max_train_items,
         args.path_to_val_map, args.max_val_items, args.sample_interval)
    #write_000_map_file(args.src_root_dir, '/data/trails/val_map_000.txt')
    #write_full_dir_map_file('/data/trails/val_map_012.txt', args.src_root_dir, test_datasets)
    print('All done.')
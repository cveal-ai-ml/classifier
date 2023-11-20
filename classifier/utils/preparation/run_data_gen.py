"""
Author: MINDFUL
Purpose: Data Organization Tool
"""

import os
import json
import shutil
import numpy as np

from tqdm import tqdm

# Template: Supervised Learning Dataset

class Dataset:

    def __init__(self, samples, labels):

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        
        return self.samples[index], self.labels[index]

    def __len__(self):

        return len(self.samples)

# Create: Folder (Override Capabilities)

def create_folder(path, override = 0):

    if(os.path.exists(path)):
        if(override):
            shutil.rmtree(path)
            os.makedirs(path) 
    else:
        os.makedirs(path) 

# Save: Basic Dataset

def save_results_basic(path_save, dataset, tag, override):

    name = "all_but_%s" % tag.lower()

    for current_key in dataset.keys():

        all_labels = dataset[current_key].labels
        all_samples = dataset[current_key].samples

        for current_label in np.unique(all_labels):

            data = [sample for sample, label in zip(all_samples, all_labels) if(label == current_label)] 

            path = os.path.join(path_save, name, current_key, str(current_label).zfill(4))
            create_folder(path, override)

            for current_file in tqdm(data, desc = "Saving %s dataset - class %s" % (current_key, current_label)):

                filename = current_file.split("/")[-1]

                shutil.copyfile(current_file, os.path.join(path, filename))

        print()

# Save: Cross Validation Dataset

def save_results_cv(path_save, cv_dataset, override):

    for current_key in cv_dataset.keys():

        all_names = ["train", "valid", "test"]

        for name in all_names:

            all_data = cv_dataset[current_key][name].samples
            all_labels = cv_dataset[current_key][name].labels

            for current_label in np.unique(all_labels):

                data = [sample for sample, label in zip(all_data, all_labels) if(label == current_label)] 
            
                path = os.path.join(path_save, "fold_%s" % str(current_key).zfill(4), name, str(current_label).zfill(4))
                create_folder(path, override)

                for current_file in tqdm(data, desc = "Saving Fold %s - %s Class %s" % (current_key, name.capitalize(), current_label)):

                    filename = current_file.split("/")[-1]

                    shutil.copyfile(current_file, os.path.join(path, filename))
                    
        print()

# Perform: Cross Validation (Supervised Dataset)

def cross_validation(dataset, k):

    num_test_samples = len(dataset.samples) // k
    num_train_samples = len(dataset.samples) - num_test_samples

    cv_dataset = {}
    
    for i in range(k):

        if(i == 0):
            j_test = [0, num_test_samples]
            j_train = [[num_test_samples, num_test_samples + num_train_samples]]
        else:
            j_test = [ele + num_test_samples for ele in j_test]

            j_train = [[0, j_test[0]]]
 
            if(j_test[1] != len(dataset.samples) - 1):
                j_train.append([j_test[1], j_test[1] + num_train_samples])
                j_train[1][1] = min(j_train[1][1], len(dataset.samples))

        # Gather testing samples, labels

        j_start, j_end = j_test
        test_samples, test_labels = dataset.samples[j_start:j_end], dataset.labels[j_start:j_end]

        # Gather training samples, labels

        train_samples, train_labels = [], []

        for j_start, j_end in j_train:
            train_samples += dataset.samples[j_start:j_end]
            train_labels += dataset.labels[j_start:j_end]

        # Gather validation samples, labels

        num_train_keep = len(train_samples) - int(len(train_samples) * 0.1)

        valid_samples, valid_labels = train_samples[num_train_keep:], train_labels[num_train_keep:]
        train_samples, train_labels = train_samples[:num_train_keep], train_labels[:num_train_keep]

        # Update cross validation dataset
        
        cv_dataset[i] = {"train": Dataset(train_samples, train_labels), 
                         "valid": Dataset(valid_samples, valid_labels),
                         "test": Dataset(test_samples, test_labels)}

    return cv_dataset  

# Perform: Oversampling With Replacement

def oversample(all_files, count, title = "Updating Data"):

    orig_size = len(all_files)

    num_to_add = count - orig_size
    
    for i in range(num_to_add):
        index = np.random.randint(0, len(all_files) - 1)
        random_file = all_files[index]
        #random_file = random_file.replace(random_file.split("/")[-1], str(orig_size + i).zfill(7) + ".png")
        all_files.append(random_file)

    return all_files

# Oversample: Minority Class
# - Assumes 2 class problem

def oversample_data(hits, fas, title):

    if(len(hits) >= len(fas)):
        return hits, oversample(fas, len(hits))
    else:
        return oversample(hits, len(fas)), fas

# Load: JSON Information

def load_meta(path):
    
    return json.load(open(path, "r"))

# Gather: Datasaet Files

def gather_files(path_data, path_meta, choice, title = "Gathering Data", tag = None):

    meta = load_meta(path_meta)

    meta_frames = meta["frameAnnotations"]

    all_files, all_groups = [], []
    for name in tqdm(meta_frames.keys(), desc = title):

        path_frame_orig = meta_frames[name]["annotations"]["image_filename"]

        if(choice == "real" or choice == "sim"):
            if(choice in path_frame_orig):
                all_files.append(os.path.join(path_data, name.strip("f") + ".png"))

                if(tag != None and tag in path_frame_orig):
                    all_groups.append("test")
                else:
                    all_groups.append("train")

        elif(choice == "both"):
            all_files.append(os.path.join(path_data, name.strip("f") + ".png"))

            if(tag != None and tag in path_frame_orig):
                all_groups.append("test")
            else:
                all_groups.append("train")
       
        else:
            raise NotImplementedError

    return all_files, all_groups

# Format: FAs & Hits, Supervised Dataset

def format_as_dataset(fa_samples, hit_samples):

    all_samples = np.asarray(fa_samples + hit_samples)
    all_labels = np.hstack([np.zeros(len(fa_samples)), np.ones(len(hit_samples))]).astype(int)

    indices = np.arange(all_samples.shape[0])
    np.random.shuffle(indices)

    all_samples = list(all_samples[indices])
    all_labels = list(all_labels[indices])

    return Dataset(all_samples, all_labels)

# Filter: Samples (Using Meta Data)

def filter_samples(all_samples, all_meta):

    train_samples, test_samples = [], []

    for sample, meta in zip(all_samples, all_meta):
       
        if("train" in meta):
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    return train_samples, test_samples

# Gather: FAs, Hits

def gather_fas_hits(paths, tag = None):

    # Load: Hits & False Alarms
    # - This just gathers the file paths

    if(tag == None):

        fa_samples = gather_files(paths["fas"]["data"], paths["fas"]["meta"], "both", "Gathering False Alarms")
        hit_samples = gather_files(paths["hits"]["data"], paths["hits"]["meta"], "real", "Gathering Hits")

        # Format: Hits (Random Oversampling)
        # - There should be near-equal hits to false alarms

        hit_samples, fa_samples = oversample_data(hit_samples, fa_samples, "Updating Hits")

        # Organize: Supervised Dataset
        # - This process also shuffles the dataset

        return format_as_dataset(fa_samples, hit_samples)

    else:

        fa_samples, fa_meta = gather_files(paths["fas"]["data"], paths["fas"]["meta"], "both", "Gathering False Alarms", tag)
        hit_samples, hit_meta = gather_files(paths["hits"]["data"], paths["hits"]["meta"], "real", "Gathering Hits", tag)

        train_fa_samples, test_fa_samples = filter_samples(fa_samples, fa_meta)
        train_hit_samples, test_hit_samples = filter_samples(hit_samples, hit_meta)

        # Format: Hits (Random Oversampling)
        # - There should be near-equal hits to false alarms

        train_hit_samples, train_fa_samples = oversample_data(train_hit_samples, train_fa_samples, "Updating Train Hits")
        #test_hit_samples, test_fa_samples = oversample_data(test_hit_samples, test_fa_samples, "Updating Test Hits")

        # Organize: Supervised Dataset
        # - This process shuffles the dataset

        train = format_as_dataset(train_fa_samples, train_hit_samples)
        test = format_as_dataset(test_fa_samples, test_hit_samples)

        return {"train": train, "valid": test, "test": test}

# Update: Data Generation Paths

def update_paths(path_data, path_results):

    path_save = os.path.join(path_results, path_data.split("/")[-1])

    path_fas_data = os.path.join(path_data, "false_alarms", "rgb")
    path_fas_meta = os.path.join(path_data, "false_alarms", "groundTruth.json")

    path_hits_data = os.path.join(path_data, "hits", "rgb")
    path_hits_meta = os.path.join(path_data, "hits", "groundTruth.json")

    return {"data": path_data, 
            "results": path_results, "save": path_save,
            "fas": {"data": path_fas_data, "meta": path_fas_meta},
            "hits": {"data": path_hits_data, "meta": path_hits_meta}}

# Run: Cross Validation Dataset Generation Procedure 
# - Takes data folder and splits into different folds, each having a train, valid, and test.
# -- The same testing dataset is shared across all cross validation folds. Thus, this assumes no pre-determined testing dataset.

def run_cv_procedure(path_data, path_results, num_folds, override):

    # Update: Paths

    paths = update_paths(path_data, path_results)
    
    # Log: Info

    print("\nOperation - Creating Cross Validation Data\n")
    print("Path Data: %s" % paths["data"])
    print("Path Results: %s" % paths["results"])
    print("Cross Validation Folds: %s\n" % num_folds)

    # Gather: FAs & Hits 

    dataset = gather_fas_hits(paths)

    # Run: Cross Validation

    cv_dataset = cross_validation(dataset, num_folds)

    print()

    # Save: Results

    save_results_cv(paths["save"], cv_dataset, override)

    print("Operation Complete\n")

# Run: Basic Dataset Generation (Train, Test)
# - This assumes there is a pre-defined testing dataset. Will train on everything else.
# -- Validation will use the testing dataset to record validation error analytics.

def run_basic_procedure(path_data, path_results, test_tag, override):

    # Update: Paths

    paths = update_paths(path_data, path_results)
    
    # Log: Info

    print("\nOperation - Training & Testing Data\n")
    print("Path Data: %s" % paths["data"])
    print("Path Results: %s" % paths["results"])
    print("Test Dataset Tag: %s\n" % test_tag)

    # Gather: FAs & Hits 

    datasets = gather_fas_hits(paths, test_tag)

    print()

    save_results_basic(paths["save"], datasets, test_tag, override)

    print("Operation Complete\n")

# Main: Parameter Configuration

if __name__ == "__main__":

    # Create: General Parameters
    
    choice = 1
    override = 1
    path_results = "/develop/results/ehd_organized"
    #path_data = "/develop/data/20230918_EUTKS_color_on_EUTKS" 
    path_data = "/develop/data/2nd_stage_on_2023_03_ChickenLittle"

    # Create: Cross Validation Paramters

    num_folds = 5

    # Create: Basic Dataset Paramterse

    test_tag = "Chicken_Little"

    # Run: Cross Validation Dataset Generation

    if(choice == 0):

        run_procedure(path_data, path_results, num_folds, override)

    # Run: Simple Dataset Generation

    elif(choice == 1):

        run_basic_procedure(path_data, path_results, test_tag, override)

    # Validate: User Input

    else:

        raise NotImplemetnedError

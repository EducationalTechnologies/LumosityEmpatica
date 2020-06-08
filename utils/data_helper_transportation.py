import numpy as np
import zipfile, os, json, re, time
import csv
from datetime import datetime
import pandas as pd
from pandas.io.json import json_normalize
import pickle


def get_samples_from_csv(sensor_csv, selfreport_csv):
    # Accerelation-magnitude:
    # np.sqrt(np.sum([acc_x**2, acc_y**2, acc_z**2]))
    pass


def create_train_test_folders(data, new_folder_location=None, train_test_ratio=0.85, to_exclude=None):
    if new_folder_location is None:
        new_folder_location = data

    # Read in csv-files (ignore those without ID)
    dataset = dict()
    for session_zipfile in os.listdir(data):
        if "ID_" not in session_zipfile:
            continue
        print("Processing zip-file: " + session_zipfile)
        archive = zipfile.ZipFile(os.path.join(data, session_zipfile), "r")
        # Get sensor file and corresponding selfreport file
        print(archive)
        filelist = archive.infolist()
        sensor_csv, selfreport_csv = None, None
        for f in filelist:
            if "_sensors" in f.filename:
                sensor_csv = csv.reader(archive.read(f))
            elif "_selfreport" in f.filename:
                selfreport_csv = csv.reader(archive.read(f))
            # NOTE it could be that there are multiple sensors and selfreports
            if sensor_csv is not None and selfreport_csv is not None:
                # TODO get data
                samples = get_samples_from_csv(sensor_csv, selfreport_csv)
                # TODO add to dataset
                # Reset then for next session (if there are more sessions in zipfile)
                sensor_csv, selfreport_csv = None, None


    # Go through sensors (in a range of 512 readings) and annotate the majority class of selfreport in that timewindow
    # ignore the sensors mentioned in `to_exclude`

    # mask with train_test_ratio*len(annotations) amount of ones
    train_mask = np.zeros(len(annotations), dtype=int)
    train_mask[:int(len(annotations)*train_test_ratio)] = 1
    np.random.shuffle(train_mask)
    train_mask = train_mask.astype(bool)

    train_annotations = annotations[train_mask]
    train_sensor_data = tensor_data[train_mask]
    test_annotations = annotations[~train_mask]
    test_sensor_data = tensor_data[~train_mask]

    ann_name = "annotations.pkl"
    sensor_name = "sensor_data.pkl"

    os.makedirs(f'{new_folder_location}/train', exist_ok=True)
    with open(f'{new_folder_location}/train/{ann_name}', "wb") as f:
        pickle.dump(train_annotations, f)
    with open(f'{new_folder_location}/train/{sensor_name}', "wb") as f:
        pickle.dump(train_sensor_data, f)

    os.makedirs(f'{new_folder_location}/test', exist_ok=True)
    with open(f'{new_folder_location}/test/{ann_name}', "wb") as f:
        pickle.dump(test_annotations, f)
    with open(f'{new_folder_location}/test/{sensor_name}', "wb") as f:
        pickle.dump(test_sensor_data, f)


def create_csv_from_MLT(mlt_file):
    # if mlt_file ends with zip: open and get _selfreport.json
    # else: assume its _selfreport.json
    # flag if we extracted the json from a zip (gets deleted afterwards)
    toDelete = False
    if "zip" in mlt_file[-4:]:
        archive = zipfile.ZipFile(mlt_file, "r")
        # Find the selfreport file
        for f in archive.infolist():
            if "selfreport" in f.filename:
                extraction_destination = os.path.join(*mlt_file.split("/")[:-1])
                archive.extract(f, path=extraction_destination)
                selfreport_file = os.path.join(extraction_destination, f.filename)
                toDelete = True
    elif "json" in mlt_file[:-4]:
        selfreport_file = mlt_file
    else:
        raise FileNotFoundError("Unkown file name")

    # create a csv with the headers
    with open(selfreport_file[:-4]+"csv", "w") as csv_file:
        csv_file.write("Time,Transportation_Mode,Status\n")
        # Time,Transportation_Mode,Status
        # 2020-06-01T10:39:30.807,Car,true
        # 2020-06-01T15:49:50.260,Car,false

        # read in json file
        with open(selfreport_file, "r") as f:
            selfreport_json = json.load(f)
        record_start_time = datetime.strptime(selfreport_json["recordingID"], "%Y-%m-%dT%H-%M-%S-%f")
        for interval in selfreport_json["intervals"]:
            start_time = datetime.strptime(interval["start"], "%H:%M:%S.%f") - datetime.strptime("00:00", "%H:%M")
            end_time = datetime.strptime(interval["end"], "%H:%M:%S.%f") - datetime.strptime("00:00", "%H:%M")
            transport_mode = interval["annotations"]["Transportation_Mode"]
            # add to CSV:
            # record_start_time+start_time,transport_mode,true
            # record_start_time+end_time,transport_mode,false
            csv_file.write((record_start_time+start_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+","+transport_mode+",true\n")
            csv_file.write((record_start_time+end_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+","+transport_mode+",false\n")
    if toDelete:
        os.remove(selfreport_file)


if __name__ == "__main__":
    create_csv_from_MLT("../manual_sessions/blackforestMLT/ID_ddm_bighike-mountain_2020-06-02T10-40-46-094_MLT.zip")
    # create_train_test_folders(data="../manual_sessions/blackforest")
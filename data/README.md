# ECG Dataset Preparation

To run the code, the dataset needs to be prepared and satisfy a couple of conditions.
- ECG signals needs to be `numpy.ndarray` and be __saved as PKL files__.
- There must be a `pandas.DataFrame` __index file which contains the information of the dataset__.
  1) A list of ECG file names.
  2) A list of ECG sampling rates if the signals have different sampling rate.
  3) A list of labels of ECGs if it is for downstream training.

We present dummy ECG signals (`./dummy/ecgs/###.pkl`) and index file (`./dummy/index.csv`) as an example.

You can build your own dataset by running `process_ecg.py`. The code reads WFDB files and saves ECG PKL files with the corresponding index CSV file which can be used when pre-training. 
```
python process_ecg.py \
    --input_dir ${DATABASE_DIRECTORY} \
    --output_dir ${WAVEFORM_DIRECTORY} \
    --index_path ${INDEX_PATH}
```
- `input_dir`: directory where the raw WFDB waveform files were saved.
- `output_dir`: directory where the ECG PKL files will be saved.
- `index_path`: Path where the index CSV file will be saved. 

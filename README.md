# VoltVision

## Pip Installs
pip install requests  
pip install bs4  
pip install pandas  
pip install lxml  
pip install statsmodels  
pip install numpy  
pip install seaborn  



## Setting Up the Data

To download the public data that is used for this model or change the source of data, run/edit the data_download.py file.

This will open a popup window which will display the progress of the files downloading and then store all downloaded files into a created data_raw file, this contains all the files in their original filename state.

Closing the window at any point will cancel the downloading of data, please note that if you stop the data download without finishing the download you will need to restart the download from the beginning.

## Converting Raw Data

As some of the files can be .XML files which are hard to parse, running xml_to_csv.py will convert all the .XML files and store them in a folder.

To run this file please use: python xml_to_csv.py <src_xml_dir> <out_csv_dir>

For example, to convert/move files from data_raw (folder created when data_download is run) to destination file data_csv, you would use:

python scripts/xml_to_csv.py ../VoltVision/data_raw ../VoltVision/data_csv

Ctrl + C at any point will cancel the conversion/movement of files.

If you do prefer to use the raw .XML files, that will still be available in the original raw data folder. However all CSVs would be moved to maintain data singularity in folders. This will also create a master CSV file for the moved XML files.


## Running the Machine Learning Model

To run the machine learning model with the data us the format:

python ../VoltVision/GenerationPredictor/main.py [-h] csv_path [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--random_state RANDOM_STATE] [--show_days SHOW_DAYS]

For example, the use the dataset Demand with a test size of 0.3 and validation size of 0.15 for a 90 day forecast the following would be used:

python ../VoltVision/GenerationPredictor/main.py data_csv/Demand_ALL.csv --test_size 0.2 --val_size 0.15 --show_days 30   

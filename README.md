# VoltVision

## Pip Installs
pip install requests\\
pip install bs4\\
pip install pandas\\
pip install lxml\\
pip install statsmodels\\
pip install numpy\\
pip install seaborn\\


## Setting Up the Data

To download the IESO public data or change the source of data, run/edit the IESO_Data_Download.py file.

This will then store all downloaded files in the ieso_raw file, this contains all the files in their original filename state.

## Converting Raw Data

As some of the files can be .XML files which are hard to parse, running XML_to_CSV_Converter.py will convert all the .XML files and store them in a folder.

To run this file please use: python XML_to_CSV_Converter.py <src_xml_dir> <out_csv_dir>

For example, to convert/move files from ieso_raw to a destination file ieso_csv, you would use:

python XML_to_CSV_Converter.py ../ieso_raw ../ieso_csv

Ctrl + C at any point will cancel the conversion/movement of files.

If you do prefer to use the raw .XML file, that will still be available in the original raw data folder. However all CSVs would be moved to maintain data singularity in folders. This will also create a master CSV file for the moved .XML files.


## Running the Machine Learning Model

To run the machine learning model with the data us the format:

python EnergyProduction.py [-h] csv_path [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--random_state RANDOM_STATE] [--show_days SHOW_DAYS]

For example, the use the dataset GenOutput_ALL with a test size of 0.3 and validation size of 0.15 for a 90 day forecast the following would be used:

python EnergyProduction.py ieso_csv\GenOutput_ALL.csv --test_size 0.2 --val_size 0.15 --show_days 30   
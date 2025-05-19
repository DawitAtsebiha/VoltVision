# VoltVision

pip install requests
pip install bs4
pip install pandas
pip install lxml

To download the IESO public data or change the source of data, run/edit the IESO_Data_Download.py file.

This will then store all downloaded files in the ieso_raw file, this contains all the files in their original filename state.

As some of the files can be .XML files which are hard to parse, running XML_to_CSV_Converter.py will convert all the .XML files and store them in a folder.

To run this file please use: python XML_to_CSV_Converter.py <src_xml_dir> <out_csv_dir>

For example, to convert/move files from ieso_raw to a destination file ieso_csv, you would use:

python XML_to_CSV_Converter.py ../ieso_raw ../ieso_csv

Ctrl + C at any point will cancel the conversion/movement of files.

If you do prefer to use the raw .XML file, that will still be available in the original raw data folder. However all CSVs would be moved to maintain data singularity in folders. This will also create a master CSV file for the moved .XML files.
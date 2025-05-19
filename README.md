# VoltVision

pip install requests
pip install bs4
pip install pandas
pip install lxml

To download the IESO public data or change the source of data, run/edit the IESO_Data_Download.py file.

This will then store all downloaded files in the ieso_raw file, this contains all the files in their original filename state.

As some of the files can be .XML files which are hard to parse, running XML_to_CSV_Converter.py will convert all the .XML files and store them in a folder.

To run this file please use: python .\XML_to_CSV_Converter.py <\Source Folder> <\Destination Folder>
## Intercity human dynamics during holiday weeks throughout the Covid-19 pandemic: A perspective of hybrid physical-virtual space

### Program Introduction
Here is a code project for calculating the hyper-adjacency matrix of the physical-virtual space. :wink:

### Program Usage Guide
#### 1. Data Preparation
**Prepare a DATA folder containing two subfolders: bandwidth and rawData.** :cat:

The bandwidth subfolder stores a CSV file that contains the distance bandwidth (λ), structured like the provided example file.

The ``DATA\rawData`` subfolder contains:

1. A city coordinates file named “city_xy.csv”, structured like the provided example file.
2. Yearly OD files for search and travel activities, with the .edges file extension, structured like the provided example file.
The naming convention for these files should be: "travel_year_normalized.edges" for normalized travel edge data of a specific year/ "search_year_normalized.edges" for normalized search edge data of a specific year.

**Subsequently, modify the ``config.py`` file and change the "ROOT_DIR" variable to the path of the aforementioned DATA folder.** :hamster:

#### 2. Program Execution
**Run the main.py file.** :full_moon_with_face:

Open the terminal and run ``main.py --time year``, where "year" is the year for which the data will be calculated, such as ``main.py --time 2020``

The hyper-adjacency matrix will be stored in ``DATA\supraMatrix``:blush:.

Last updated on 2024/10/15. Have a nice day!


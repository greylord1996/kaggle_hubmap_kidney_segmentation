# kaggle_hubmap_kidney_segmentation

### Silver medal solution (55th place over 1200 teams) for https://www.kaggle.com/c/hubmap-kidney-segmentation/ challenge

## Installation

First of all, you should have python 3.x to work with this project. The recommended Python version is 3.6 or greater.

Note for Windows users: You should start a command line with administrator's privileges.

First of all, clone the repository:

    git clone https://github.com/greylord1996/kaggle_hubmap_kidney_segmentation.git
    cd kaggle_hubmap_kidney_segmentation/

Create a new virtual environment:

    # on Linux:
    python -m venv hubmapenv
    # on Windows:
    python -m venv hubmapenv

Activate the environment:

    # on Linux:
    source hubmapenv/bin/activate
    # on Windows:
    call hubmapenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install -r requirements.txt
    # on Windows:
    python -m pip install -r requirements.txt


## Data

To use this code you need to download data:

- Kaggle input data: https://www.kaggle.com/c/hubmap-kidney-segmentation/data
- 512x512 tiles data: https://www.kaggle.com/iafoss/hubmap-512x512


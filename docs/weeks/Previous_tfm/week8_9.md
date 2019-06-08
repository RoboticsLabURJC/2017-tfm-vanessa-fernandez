---
layout: default
---
# Weeks 8,9: State of the art, dataset of cultivations, and installation od DetectionSuite

These weeks, I have made the state of the art of Deep Learning for object detection. The state of the art is available in the following [link](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/tree/master/State%20of%20the%20art). 


Also, this week I've been looking for information on the crop dataset. Some datasets are: 

* [FAOSTAT](http://www.fao.org/faostat/en/#home): provides free access to food and agriculture data for over 245 countries and territories and covers all FAO regional groupings from 1961 to the most recent year available.

* [Covertype Data Set](http://archive.ics.uci.edu/ml/datasets/Covertype): includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4).

* [Plants Data Set](http://archive.ics.uci.edu/ml/datasets/Plants]): Data has been extracted from the USDA plants database. It contains all plants (species and genera) in the database and the states of USA and Canada where they occur.

* [Urban Land Cover Data Set](http://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover): Contains training and testing data for classifying a high resolution aerial image into 9 types of urban land cover. Multi-scale spectral, size, shape, and texture information are used for classification. There are a low number of training samples for each class (14-30) and a high number of classification variables (148), so it may be an interesting data set for testing feature selection methods. The testing data set is from a random sampling of the image. Class is the target classification variable. The land cover classes are: trees, grass, soil, concrete, asphalt, buildings, cars, pools, shadows.

* [Area, Yield and Production of Crops]([https://data.gov.ie/dataset/area-yield-and-production-of-crops): Area under crops, yield of crops in tonnes per hectare, and production of crops in tonnes, classified by type of crop.

* [Target 8 Project cultivation database](https://plantnetwork.org/plantnetwork-projects/target-8-project-cultivation-database/): has been written in Microsoft Access, and provides a valuable source of information on habitats, ecology, reproductive attributes, germination protocol and so forth (see below). Attributes can be compared, and species from similar habitats can be quickly listed. This may help you in achieving the right conditions for a species, by utilising your knowledge of more familiar plants. The target 8 dataset form presents the data for the project species only, with a sub-form giving a list of submitted protocols. A full dataset to the British and Irish flora is also avaialble (Entire dataset form) which gives the same ata for not threatened taxa as well as many neophytes.

* [Datasets of Nelson Institute University of Wisconsin-Madison](https://nelson.wisc.edu/sage/data-and-models/datasets.php): There are some datasets for crop or vegetation.

* [Global data set of monthly irrigated and rainfed crop areas around the year 2000 (MIRCA2000)](https://www.uni-frankfurt.de/45218023/MIRCA) [2](https://www.uni-frankfurt.de/45217790/FHP_06_Portmann_et_al_2008.pdf): is a global dataset with spatial resolution of 5 arc minutes (about 9.2 km at the equator) which provides both irrigated and rainfed crop areas of 26 crop classes for each month of the year. The crops include all major food crops (wheat, maize, rice, barley, rye, millet, sorghum, soybean, sunflower, potato, cassava, sugarcane, sugarbeet, oil palm, canola, groundnut, pulses, citrus, date palm, grape, cocoa, coffee, other perennials, fodder grasses, other annuals) as well as cotton. For some crops no adequate statistical information was available to retain them individually and these crops were grouped into ‘other’ categories (perennial, annual and fodder grasses).

* [Root crops and plants harvested green from arable land by area](https://data.europa.eu/euodp/es/data/dataset/1X3d9bfhzwGDJQ66RYHoSQ): includes the areas where are sown potatoes (including seed), sugar beet (excluding seed), temporary grasses for grazing, hay or silage (in the crop rotation cycle), leguminous plants grown and harvested green (as the whole plant, mainly for fodder, energy or green manuring use), other cereals harvested green (excluding green maize): rye, wheat, triticale, annual sorghum, buckwheat, etc. This indicator uses the concepts of "area under cultivation" which corresponds: • before the harvest, to the sown area; • after the harvest, to the sown area excluding the non-harvested area (e.g. area ruined by natural disasters, area not harvested for economic reasons, etc.).


Also, I installed DetectionSuite. I have followed the steps indicated in the following [link](https://github.com/JdeRobot/dl-DetectionSuite). 
It is important to install version 9 of CUDA, because other versions will probably give an error.


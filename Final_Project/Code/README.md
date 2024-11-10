# Renewable Energy Datasets for Research and Analysis

This repository provides access to two comprehensive datasets focused on renewable energy generation, designed to support advanced research in areas such as energy forecasting, grid reliability assessment, and economic modeling. These datasets contain detailed information on wind power generation, operational characteristics, and other renewable energy input variables, offering extensive resources for studies in renewable energy management and technology innovation.

---

## 1. U.S. Wind Power Generation Dataset

This dataset offers an extensive hourly view of wind power generation across various wind plants in the United States, spanning the years 1980 through 2022. It includes detailed metadata about each plant, including geographic coordinates, turbine specifications, and system capacity.

- **Applications**:
  - Renewable energy forecasting
  - Grid reliability assessment
  - Economic modeling
  - Regional wind pattern analysis
  - Weather impact evaluation on energy production

- **Key Features**:
  - Detailed metadata for wind plants
  - Geographic and operational attributes
  - Hourly generation data for individual turbines
  - Wind turbine power curve details

- **Data Access**: Available as open-access on [Zenodo](https://zenodo.org/records/8240163).

- **Related Publication**: [Scientific Data Article](https://www.nature.com/articles/s41597-024-03894-w#code-availability).

---

## 2. Renewable Energy Generation Input Feature Variables Dataset

This dataset supports research into renewable energy input features and their impact on energy generation. It includes variables influencing energy production from wind, solar, and other renewable sources, providing a basis for evaluating and improving energy forecasting models.

- **Applications**:
  - Input feature analysis for renewable energy generation
  - Renewable energy resource modeling
  - Sensitivity analysis of environmental factors on generation

- **Key Features**:
  - Analysis of input variables for renewable energy generation
  - Focus on wind, solar, and other energy sources
  - Rich feature set for renewable energy resource evaluation

- **Data Access**: Available in the [GitHub Repository](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/tree/main).

- **Related Publication**: [Scientific Data Article](https://www.nature.com/articles/s41597-022-01696-6?fromPaywallRec=false#Sec3).

---

## Data Dictionary

The following table provides a detailed description of the variables included in the **U.S. Wind Power Generation Dataset**:

| Variable Name                     | Definition                                                  | Unit       | Example Values                              |
|-----------------------------------|-------------------------------------------------------------|------------|--------------------------------------------|
| `plant_code`                      | Unique identifier for each wind plant                       | ID         | 508, 692, 944                              |
| `plant_code_unique`               | Unique identifier for individual turbines in a plant        | ID         | 508_1, 692_1                               |
| `generator_id`                    | Identifier for specific generators                          | ID         | T1, T2, T3                                |
| `lat`                             | Latitude of the plant                                       | Degrees    | 38.033327                                  |
| `lon`                             | Longitude of the plant                                      | Degrees    | -102.537915                                |
| `ba`                              | Balancing Authority controlling the plant                   | Abbrev.    | PSCO, WAUW                                 |
| `nerc_region`                     | NERC region where the plant is located                      | Abbrev.    | WECC, MRO                                 |
| `state`                           | State where the plant is located                            | Abbrev.    | CO, WY, IL                                |
| `system_capacity`                 | Maximum power output capacity                               | MW         | 1500.0                                     |
| `wind_farm_xCoordinates`          | X-coordinates of wind turbines within the farm              | Meters     | [0, 660.0, 330.0]                         |
| `wind_farm_yCoordinates`          | Y-coordinates of wind turbines within the farm              | Meters     | [0, 376.0, 752.0]                         |
| `wind_turbine_hub_ht`             | Hub height of the turbine                                   | Meters     | 79.98                                      |
| `wind_turbine_powercurve_powerout`| Power output for each wind speed in the power curve         | kW         | [0.0, 0.0, 0.0]                           |
| `wind_turbine_powercurve_windspeeds`| Wind speeds corresponding to the power curve               | m/s        | [0.0, 0.25, 0.5]                          |
| `wind_turbine_rotor_diameter`     | Diameter of the turbine's rotor                             | Meters     | 82.5                                       |
| `wind_resource_shear`             | Shear factor describing wind speed variation with height    | N/A        | 0.14                                       |
| `wind_resource_turbulence_coeff`  | Turbulence coefficient of the wind resource                 | N/A        | 0.1                                        |
| `wind_resource_model_choice`      | Wind resource model used for simulations                    | Categorical| 0, 1, 2                                   |
| `wind_farm_wake_model`            | Model used to simulate wake effects within the wind farm    | Categorical| 0, 1                                      |
| `turb_generic_loss`               | Generic loss coefficient for the turbine                    | Percent    | 15                                         |
| `adjust:constant`                 | Adjustment constant applied to the generation data          | N/A        | 0                                          |

For the **Renewable Energy Generation Input Feature Variables Dataset**, refer to the [GitHub Repository](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/tree/main) for variable descriptions.

---

## How to Use These Datasets

1. Access the data through the provided links to Zenodo and GitHub.
2. Use the datasets for studies on renewable energy forecasting, technology assessment, and grid management.
3. Consult the referenced publications for insights into dataset structure and methodologies.


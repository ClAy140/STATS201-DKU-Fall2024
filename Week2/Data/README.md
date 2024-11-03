# U.S. Wind Power Generation Dataset

This dataset provides a comprehensive view of hourly wind power generation across various wind plants in the United States, covering data from 1980 through 2022. It includes essential details on each plant, such as geographic coordinates, system capacity, turbine specifications, and other operational characteristics. The dataset is structured to support research and analysis in renewable energy forecasting, grid reliability assessment, and economic modeling. By examining hourly generation data and plant metadata, researchers can analyze regional wind patterns, understand the impact of weather conditions on energy production, and optimize renewable energy integration into the power grid.

Available as open-access on [Zenodo](https://zenodo.org/records/8240163). This dataset is ideal for studies in fields like energy modeling, renewable resource management, and technology innovation assessment in wind energy.

## Data Dictionary

| Variable Name                     | Definition                                                  | Unit      | Range or Example               | Sample Observation |
|-----------------------------------|-------------------------------------------------------------|-----------|--------------------------------|---------------------|
| `plant_code`                      | Unique identifier for each wind plant                       | ID        | 508, 692, 944                 | 508                 |
| `plant_code_unique`               | Unique identifier for individual turbines in a plant        | ID        | 508_1, 692_1                  | 508_1               |
| `generator_id`                    | Identifier for specific generators                          | ID        | T1, T2, T3                    | T4                  |
| `lat`                              | Latitude of the plant                                       | Degrees   | 25.0 - 50.0                   | 38.033327           |
| `lon`                              | Longitude of the plant                                      | Degrees   | -125.0 - -70.0                | -102.537915         |
| `ba`                               | Balancing Authority controlling the plant                   | Abbrev.   | PSCO, WAUW                    | PSCO                |
| `nerc_region`                      | NERC region where the plant is located                      | Abbrev.   | WECC, MRO                     | WECC                |
| `state`                            | State where the plant is located                            | Abbrev.   | CO, WY, IL                    | CO                  |
| `system_capacity`                  | Maximum power output capacity                               | MW        | 0.1 - 200                     | 1500.0              |
| `wind_farm_xCoordinates`           | X-coordinates of wind turbines within the farm              | Meters    | [0, 660.0, 330.0]             | [0, 660.0, 330.0]   |
| `wind_farm_yCoordinates`           | Y-coordinates of wind turbines within the farm              | Meters    | [0, 376.0, 752.0]             | [0, 376.0, 752.0]   |
| `wind_turbine_hub_ht`              | Hub height of the turbine                                   | Meters    | 30 - 100                      | 79.98               |
| `wind_turbine_powercurve_powerout` | Power output for each wind speed in the power curve         | kW        | [0, 50, 100, ..., 1500]       | [0.0, 0.0, 0.0]     |
| `wind_turbine_powercurve_windspeeds` | Wind speeds corresponding to the power curve                | m/s       | [0, 2, 4, ..., 25]            | [0.0, 0.25, 0.5]    |
| `wind_turbine_rotor_diameter`      | Diameter of the turbine's rotor                             | Meters    | 30 - 100                      | 82.5                |
| `wind_resource_shear`              | Shear factor describing wind speed variation with height    | N/A       | 0.1 - 0.2                     | 0.14                |
| `wind_resource_turbulence_coeff`   | Turbulence coefficient of the wind resource                 | N/A       | 0.05 - 0.15                   | 0.1                 |
| `wind_resource_model_choice`       | Wind resource model used for simulations                    | Categorical | 0, 1, 2                     | 0                   |
| `wind_farm_wake_model`             | Model used to simulate wake effects within the wind farm    | Categorical | 0, 1                         | 0                   |
| `turb_generic_loss`                | Generic loss coefficient for the turbine                    | Percent   | 0 - 20                        | 15                  |
| `adjust:constant`                  | Adjustment constant applied to the generation data          | N/A       | -                             | 0                   |

## Links
- **Dataset**: [U.S. Wind Power Generation Dataset on Zenodo](https://zenodo.org/records/8240163)
- **Publication**: [Scientific Data Article](https://www.nature.com/articles/s41597-024-03894-w#code-availability)

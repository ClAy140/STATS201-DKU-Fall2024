# Wind Power Forecasting Using Transfer Learning

## Overview
This project explores the application of **Transfer Learning** to improve wind power forecasting across different regions with varying data characteristics. Traditional wind power forecasting models trained on specific geographic locations often struggle to generalize to other regions due to **climatic variability**, differing **data resolutions**, and the **availability of meteorological features**. This study applies transfer learning to address these issues by training a model on a data-rich region (**China**) and applying it to a data-scarce region (**U.S.**), significantly enhancing forecasting accuracy.

## Key Concepts
- **Transfer Learning**: Leveraging knowledge from one region (e.g., China) to improve forecasting in another region (e.g., U.S.) with limited data.
- **Climatic Variability**: Differences in weather patterns, such as wind speed and temperature, which affect wind power generation and forecasting accuracy.
- **LSTM Networks**: A type of **Recurrent Neural Network (RNN)** used for time-series forecasting, well-suited for capturing long-term dependencies in wind power data.

## Project Structure

### `Code/`
Contains the main scripts and helper files for data processing, model training, and evaluation.

- `Explanation.py`: Provides detailed explanations of the model's architecture and methodology.
- `glance.py`: Contains functions to visualize key metrics and insights from the data.
- `main.py`: Main script to run the transfer learning model for forecasting wind power.
- `pre.py`: Data preprocessing script for cleaning and preparing datasets.
- `predict_china.py`: Script for predicting wind power generation using the trained model on the Chinese dataset.
- `Wind_capacity_99MW.csv`: Dataset containing the wind plant capacities.
- `china_jan_2020.csv`: Raw meteorological data for China (January 2020).
- `china_jan_2020.pkl`: Preprocessed and saved version of the China dataset.
- `china_jan_2020_o.pkl`: Another version of the preprocessed China dataset.
- `eia_wind_configs.csv`: EIA configurations for wind plants.
- `us_jan_2020.csv`: Raw meteorological data for the U.S. (January 2020).
- `us_jan_2020.pkl`: Preprocessed and saved version of the U.S. dataset.
- `wind_gen_cf_2020.csv`: Wind generation data for 2020.
- `output_plots/`: Folder containing generated plots and visualizations, such as heatmaps, generation patterns, and wind plant distributions.

### `Data/`
Contains raw and preprocessed datasets used in this project.

- `china_jan_2020.csv`: Raw dataset for China wind power data.
- `us_jan_2020.csv`: Raw dataset for U.S. wind power data.
- `Wind_capacity_99MW.csv`: Wind plant capacity data.
- `wind_gen_cf_2020.csv`: Wind generation capacity factors for 2020.
- `eia_wind_configs.csv`: EIA wind configurations.

### `Explanation.ipynb`
An interactive Jupyter notebook explaining the methodology, data processing, and model training process in detail.

### `Poster.pdf`
The final poster summarizing the research findings and methodology.

### `System_Configuration.py`
A Python script used to check the system configuration and required software versions.

### `installed_packages.txt`
A text file listing all the installed Python packages and dependencies.

### `system_configuration_report.txt`
A system configuration report with details of the software and hardware used in the project.

## Requirements

To run the code in this repository, you will need to install the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `keras`
- `jupyter`

You can install these packages using `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies listed in `installed_packages.txt`.

## Setup

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/transfer-learning-wind-power.git
```

2. Navigate to the `Code/` directory to run the scripts.

3. Ensure that the necessary datasets (e.g., `china_jan_2020.csv`, `us_jan_2020.csv`) are available in the `Data/` directory.

4. Run `main.py` to train the model using transfer learning.

```bash
python main.py
```

5. You can visualize the results in the `output_plots/` folder. 

## Evaluation

The model performance is evaluated using the following metrics:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

These metrics are used to assess the accuracy of the transfer learning model compared to traditional region-specific models.

## Conclusion

This project demonstrates how **transfer learning** can be applied to improve **wind power forecasting** across regions with varying data availability. By leveraging high-resolution datasets from regions like **China**, we can significantly enhance forecasting accuracy for **data-scarce regions** like the **U.S.**, leading to improved **grid stability** and supporting **decarbonization** efforts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


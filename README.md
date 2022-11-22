# Time-series prediction with Pytorch deep learning model
### Predict time series data using deep learning algorithm with Pytorch

## About this project
Nowadays data plays an important role to give insights for decision-making and even predict future outcomes. Although there are a lot of algorithms used to find correlations among features to predict target results, the most popular buzzword that you might hear these days to solve this problem is machine learning, especially deep learning. However, the deep learning model is very hard to understand and yet difficult to implement. Hence, I decided to create this project to provide a tool for time-series data prediction using the deep learning model. This project is written in Python and uses Pytorch's neural network module to create the deep learning model. 

## How it works?
For the time-series data, we usually use RNN (Recurrent neural network) for the prediction. In this project, we prepared 3 types of RNN models such as simple RNN, LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit). The model will predict data in the future state based on previous states as shown in the figure below.

![dataframe](https://github.com/supat-roong/deep_learning_time_series_prediction_pytorch/blob/main/assets/img/concept_img_1.png?raw=true)

![model](https://github.com/supat-roong/deep_learning_time_series_prediction_pytorch/blob/main/assets/img/concept_img_2.png?raw=true)
## How to use?
### Instruction on how to run this project:
1. Clone the project from [GitHub](https://github.com/supat-roong/deep_learning_time_series_prediction_pytorch.git)
    ```
    $ git clone https://github.com/supat-roong/deep_learning_time_series_prediction_pytorch.git
    ```
2. Change the working directory to root
    ```
    $ cd deep_learning_time_series_prediction_pytorch
    ```
3. Create a virtual environment (optional)
    ```
    $ conda create --name <env_name> --file requirements.txt
    ```
    then activate the environment
    ```
    $ conda activate <env_name>
    ```
4. Download the dependencies from `requirement.txt`
    ```
    $ pip install -r requirements.txt
    ```
5. Place your dataset in `assets/data/`
6. Setup config file corresponding to your data in `assets/config/config.json`
7. Run the main program
    ```
    $ python main.py -dn <your_dataset_file_name>
    ```

    When running the main program you can config your setup using parser arguments below

    | Parser Flag | Argument Name | Description    |
    | :---:       |    :----:   |          :--- |
    | -df         | data_folder       | data folder path   |
    | -dn         | data_file_name        | data file name  |
    | -cf         | config_folder       | config folder path  |
    | -cn         | config_file_name       | config file name   |
    | -of         | ouput_folder       | output folder path   |
    | -pn         | prediction_data_file_name       | prediction data file name   |
    | -gn    | graph_file_name       | output graph data file name   |
    | -sd    | do_save_output_data       | flag to save output prediction data to csv file   |
    | -vi    | do_visualize_input_data       | flag to only visualize input data (no model training)   |
    | -cr    | do_not_compare_result       | flag to disable ground truth and prediction result compare   |

### Example
You can check if the program is running properly with this example
```
$ python main.py -dn demo_dataset_tetuan_city_power_consumption.csv
```
### Expected output
1. Terminal output
   - Training information in the form of epoch and loss
   - Process time when finish
2. File output
   - Image files of the prediction graph in pdf and png format
     - Example of output graph from previous example
     ![output graph](https://github.com/supat-roong/deep_learning_time_series_prediction_pytorch/blob/main/assets/img/demo_output_graph.png?raw=true)

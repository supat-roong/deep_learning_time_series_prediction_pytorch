import pandas as pd
import json
from data_preparation import *
from data_visualization import *
from model_selection import ModelSelector
from model_training import ModelTrainer


class DeepLearningPredictor:
    def __init__(
        self,
        input_data_file_path,
        config_file_path,
        output_data_file_path,
        output_graph_file_path,
        do_save_output_data,
        do_visualize_input_data,
        do_compare_result,
    ):
        # Declare parameters
        self.input_data_file_path = input_data_file_path
        self.output_data_file_path = output_data_file_path
        self.output_graph_file_path = output_graph_file_path
        self.do_save_output_data = do_save_output_data
        self.do_visualize_input_data = do_visualize_input_data
        self.do_compare_result = do_compare_result

        # Load config map
        config_map = json.load(open(config_file_path))

        # Unpack config map
        # Data preparation
        data_preparation_config_map = config_map["data_preparation"]
        # Declare data preparation parameters
        self.seq_length = data_preparation_config_map["seq_length"]
        self.train_set_ratio = data_preparation_config_map["train_set_ratio"]

        # Model selection
        self.model_selection_config_map = config_map["model_selection"]

        # Model training
        self.model_training_config_map = config_map["model_training"]

        # Data visualization
        data_visualization_config_map = config_map["data_visualization"]
        self.input_data_plot_title = data_visualization_config_map[
            "input_data_plot_title"
        ]
        self.prediction_data_plot_title = data_visualization_config_map[
            "prediction_data_plot_title"
        ]

    def __call__(self):
        """
        Main program including read data, prepare data, select model, 
        train model, prediction, data visualization, and write prediction data
        """
        # Read data and column name from data file
        column_name_list, data = self.read_data(self.input_data_file_path)

        # Visualize input data
        if self.do_visualize_input_data:
            plot_input_data(
                column_name_list=column_name_list,
                input_data=data,
                plot_title=self.input_data_plot_title,
                output_image_name=self.output_graph_file_path,
            )
            return

        # Prepare data
        (
            data_x,
            data_y,
            train_x,
            train_y,
            test_x,
            test_y,
            train_size,
            scaler,
        ) = prepare_data(data, self.seq_length, self.train_set_ratio)

        # Select model
        model = ModelSelector(self.model_selection_config_map).create_model()

        # Train model
        model_trainer = ModelTrainer(self.model_training_config_map)
        trained_model = model_trainer.train_model(model, train_x, train_y)

        # Prediction
        predict_y = model_trainer.predict(trained_model, data_x)

        # Denomalization
        predict_y_plot = scaler.inverse_transform(predict_y.data.numpy())
        data_y_plot = scaler.inverse_transform(data_y.data.numpy())

        # Data visualization
        plot_prediction(
            column_name_list=column_name_list,
            input_data=data_y_plot,
            prediction_data=predict_y_plot,
            train_size=train_size,
            plot_title=self.prediction_data_plot_title,
            output_image_name=self.output_graph_file_path,
            do_compare_result=self.do_compare_result,
        )

        # Save prediction data
        if self.do_save_output_data:
            self.write_data(
                column_name_list, predict_y_plot, self.output_data_file_path
            )

    def read_data(self, input_data_file_path):
        """
        Read data from .csv file to pd.Dataframe

        Args:
            input_data_file_path (str)

        Returns:
            column_name_list (np.array)
            data (np.array)
        """
        dataframe = pd.read_csv(input_data_file_path)
        column_name_list = dataframe.columns.to_numpy()
        data = dataframe.values
        return column_name_list, data

    def write_data(self, column_name_list, data, output_data_file_path):
        """
        Write data to .csv file

        Args:
            column_name_list (np.array)
            data (np.array)
            output_data_file_path (str)
        """
        dataframe = pd.DataFrame(data, columns=column_name_list)
        dataframe.to_csv(output_data_file_path, index=False)

import os
import time
import argparse
from deep_learning_predictor import DeepLearningPredictor


def main():
    """
    Recieve variables value from parser, then input to the main deep learning predictor class object and call prediction
    """
    # Start time
    start = time.time()
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-df",
        "--data_folder",
        type=str,
        default=os.path.join("assets", "data"),
        help="data folder path",
    )
    parser.add_argument(
        "-dn", "--data_file_name", type=str, required=True, help="data file name"
    )
    parser.add_argument(
        "-cf",
        "--config_folder",
        type=str,
        default=os.path.join("assets", "config"),
        help="config folder path",
    )
    parser.add_argument(
        "-cn",
        "--config_file_name",
        type=str,
        default="config.json",
        help="config file name",
    )
    parser.add_argument(
        "-of",
        "--ouput_folder",
        type=str,
        default=os.path.join("assets", "output"),
        help="output folder path",
    )
    parser.add_argument(
        "-pn",
        "--prediction_data_file_name",
        type=str,
        default="predicted_data.csv",
        help="prediction data file name",
    )
    parser.add_argument(
        "-gn",
        "--graph_file_name",
        type=str,
        default="output_graph",
        help="output graph data file name",
    )
    parser.add_argument(
        "-sd",
        "--do_save_output_data",
        action="store_true",
        help="flag to save output prediction data to csv file",
    )
    parser.add_argument(
        "-vi",
        "--do_visualize_input_data",
        action="store_true",
        help="flag to only visualize input data (no model training)",
    )
    parser.add_argument(
        "-cr",
        "--do_not_compare_result",
        action="store_false",
        help="flag to disable ground truth and prediction result compare",
    )

    args = parser.parse_args()

    # Declare variables from parser
    data_folder = args.data_folder
    data_file_name = args.data_file_name
    config_folder = args.config_folder
    config_file_name = args.config_file_name
    ouput_folder = args.ouput_folder
    prediction_data_file_name = args.prediction_data_file_name
    graph_file_name = args.graph_file_name
    do_save_output_data = args.do_save_output_data
    do_visualize_input_data = args.do_visualize_input_data
    do_compare_result = args.do_not_compare_result

    # Join file path
    input_data_file_path = os.path.join(data_folder, data_file_name)
    config_file_path = os.path.join(config_folder, config_file_name)
    output_data_file_path = os.path.join(ouput_folder, prediction_data_file_name)
    output_graph_file_path = os.path.join(ouput_folder, graph_file_name)

    # Create main deep learning predictor class object
    deep_learning_predictor = DeepLearningPredictor(
        input_data_file_path=input_data_file_path,
        config_file_path=config_file_path,
        output_data_file_path=output_data_file_path,
        output_graph_file_path=output_graph_file_path,
        do_save_output_data=do_save_output_data,
        do_visualize_input_data=do_visualize_input_data,
        do_compare_result=do_compare_result,
    )

    # Call main function
    deep_learning_predictor()

    # End time
    end = time.time()

    # Print process time when done
    print("Process time: {:.4f} s".format(end - start))


# Run program
if __name__ == "__main__":
    main()

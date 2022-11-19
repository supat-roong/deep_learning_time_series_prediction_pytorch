import os
import matplotlib.pyplot as plt
import numpy as np


def plot_input_data(column_name_list, input_data, plot_title, output_image_name):
    """
    Visualize input data to time-series graph by column

    Args:
        column_name_list (np.array)
        input_data (np.array): array of input data corresponding to column_name_list
        plot_title (str)
        output_image_name (str)
    """
    num_plot, num_col, num_row = get_plot_dimension(column_name_list)
    # Initiate subplot
    fig, axs = plt.subplots(num_row, num_col)
    plt.suptitle(plot_title)
    column_name_matrix = np.resize(column_name_list, (num_row, num_col))

    # Plot subplot for each column
    for i in range(num_row):
        for j in range(num_col):
            count_num_plot = num_col * i + j
            axs[i, j].plot(input_data[:, count_num_plot])
            axs[i, j].set_title(column_name_matrix[i, j])

            # Check if finish plotting
            if (count_num_plot + 1) == num_plot:
                # Remove unplotted axis
                while j < (num_col - 1):
                    j += 1
                    axs[i, j].axis("off")

                # Remove inner axis label
                for ax in axs.flat:
                    ax.label_outer()

                # Save graph
                fig.savefig(f"{output_image_name}.pdf", bbox_inches="tight")
                # Convert to png
                os.system(
                    f"pdftoppm -png -r 300 -singlefile {output_image_name}.pdf {output_image_name}"
                )
                return


def plot_prediction(
    column_name_list,
    input_data,
    prediction_data,
    train_size,
    plot_title,
    output_image_name,
    do_compare_result,
):
    """
    Visualize prediction data to time-series graph

    Args:
        column_name_list (np.array)
        input_data (np.array): array of input data corresponding to column_name_list
        prediction_data (np.array): array of prediction data
        train_size (int): training size
        plot_title (str)
        output_image_name (str)
        do_compare_result (bool): flag to also plot input data for comparison
    """
    # Check compare result flag
    if do_compare_result:
        num_plot, num_col, num_row = get_plot_dimension(column_name_list)
        # Initiate subplot
        fig, axs = plt.subplots(num_row, num_col)
        plt.suptitle(plot_title)
        column_name_matrix = np.resize(column_name_list, (num_row, num_col))

        # Plot subplot for each column
        for i in range(num_row):
            for j in range(num_col):
                count_num_plot = num_col * i + j
                axs[i, j].plot(input_data[:, count_num_plot])
                axs[i, j].plot(prediction_data[:, count_num_plot])
                axs[i, j].set_title(column_name_matrix[i, j])
                axs[i, j].axvline(x=train_size, c="r", linestyle="--")

                # Check if finish plotting
                if (count_num_plot + 1) == num_plot:
                    # Remove unplotted axis
                    while j < (num_col - 1):
                        j += 1
                        axs[i, j].axis("off")

                    # Remove inner axis label
                    for ax in axs.flat:
                        ax.label_outer()

                    # Add legend
                    fig.legend(
                        ["Ground Truth", "Prediction"],
                        loc="lower center",
                        bbox_to_anchor=(0, -0.05, 1, 1),
                        bbox_transform=plt.gcf().transFigure,
                        ncol=2,
                    )
                    # Save graph
                    fig.savefig(f"{output_image_name}.pdf", bbox_inches="tight")
                    # Convert to png
                    os.system(
                        f"pdftoppm -png -r 300 -singlefile {output_image_name}.pdf {output_image_name}"
                    )
                    return

    else:
        plot_input_data(
            column_name_list, prediction_data, plot_title, output_image_name
        )


def get_plot_dimension(column_name_list):
    """
    Find subplot dimension

    Args:
        column_name_list (np.array)

    Returns:
        tuple: num_plot, num_col, num_row
    """
    num_plot = len(column_name_list)
    num_col = int(np.ceil(np.sqrt(num_plot)))
    num_row = int(np.ceil(num_plot / num_col))
    return num_plot, num_col, num_row

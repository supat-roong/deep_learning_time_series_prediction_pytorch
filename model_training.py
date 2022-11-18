import torch


class ModelTrainer:
    def __init__(self, model_training_config_map):
        """
        Model trainer class object from config map

        Args:
            model_training_config_map (dict)
        """
        self.num_epochs = model_training_config_map["num_epochs"]
        self.learning_rate = model_training_config_map["learning_rate"]
        self.print_loss_every = model_training_config_map["print_loss_every"]

    def train_model(self, model, train_x, train_y):
        """
        Train model with MSE loss and Adam optimizer

        Args:
            model (nn.Module): deep learning model
            train_x (tensor)
            train_y (tensor)

        Returns:
            model: trained_model
        """
        model.train()
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            outputs = model(train_x)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, train_y)

            loss.backward()

            optimizer.step()

            if epoch % self.print_loss_every == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        return model

    def predict(self, model, data_x):
        """
        Predict y from input x using deep learning model

        Args:
            model (nn.Module): deep learning model
            data_x (tensor)

        Returns:
            data_y (tensor)
        """
        model.eval()
        return model(data_x)

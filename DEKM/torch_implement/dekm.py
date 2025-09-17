import torch
import torch.nn as nn


class DEKMDenseModel(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(DEKMDenseModel, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, hidden_units),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, input_shape),
        )

        # Initialize weights uniformly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return torch.cat([h, y], dim=1)


class DEKMConvModel(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(DEKMConvModel, self).__init__()
        self.input_shape = input_shape  # (H, W, C) format like TensorFlow
        self.hidden_units = hidden_units

        # Convert TensorFlow shape (H, W, C) to PyTorch shape (C, H, W)
        self.input_channels = input_shape[2]
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]

        # Filter sizes from TensorFlow model
        self.filters = [32, 64, 128, hidden_units]

        # Determine padding for third conv layer based on input shape
        if self.input_height % 8 == 0:
            self.pad3 = 1  # "same" padding for kernel_size=3, stride=2
        else:
            self.pad3 = 0  # "valid" padding

        # Calculate the spatial dimensions after encoder
        self.encoded_height = self.input_height // 8
        self.encoded_width = self.input_width // 8

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.filters[0],
            kernel_size=5,
            stride=2,
            padding=2,  # "same" padding for kernel_size=5, stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.filters[0],
            out_channels=self.filters[1],
            kernel_size=5,
            stride=2,
            padding=2,  # "same" padding for kernel_size=5, stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.filters[1],
            out_channels=self.filters[2],
            kernel_size=3,
            stride=2,
            padding=self.pad3,
        )

        # Calculate flattened size after conv3
        conv3_out_h = (
            (self.input_height // 4 + 2 * self.pad3 - 3) // 2 + 1
            if self.pad3 == 0
            else self.encoded_height
        )
        conv3_out_w = (
            (self.input_width // 4 + 2 * self.pad3 - 3) // 2 + 1
            if self.pad3 == 0
            else self.encoded_width
        )
        self.flattened_size = self.filters[2] * conv3_out_h * conv3_out_w

        # Dense layer for embedding
        self.dense_embed = nn.Linear(self.flattened_size, self.filters[3])

        # Decoder layers
        self.dense_decode = nn.Linear(
            self.filters[3], self.filters[2] * self.encoded_height * self.encoded_width
        )

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.filters[2],
            out_channels=self.filters[1],
            kernel_size=3,
            stride=2,
            padding=self.pad3,
            output_padding=1 if self.pad3 == 1 else 0,
        )

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=self.filters[1],
            out_channels=self.filters[0],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=self.filters[0],
            out_channels=self.input_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        # Activation function
        self.relu = nn.ReLU()

        # Initialize weights uniformly like TensorFlow
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights uniformly to match TensorFlow initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input x should be in (N, C, H, W) format (PyTorch convention)
        # If input is in (N, H, W, C) format, convert it
        if x.dim() == 4 and x.shape[1] == self.input_height:
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten and get embedding
        x_flat = x.view(x.size(0), -1)
        h = self.dense_embed(x_flat)  # This is the embedding/encoding

        # Decoder
        x_decode = self.relu(self.dense_decode(h))
        x_decode = x_decode.view(
            x_decode.size(0), self.filters[2], self.encoded_height, self.encoded_width
        )

        x_decode = self.relu(self.deconv1(x_decode))
        x_decode = self.relu(self.deconv2(x_decode))
        x_decode = self.deconv3(x_decode)  # No activation on final layer

        # Flatten the decoded output
        x_decode_flat = x_decode.view(x_decode.size(0), -1)

        # Concatenate embedding and flattened reconstruction
        # This matches the TensorFlow model: layers.Concatenate()([h, layers.Flatten()(x)])
        output = torch.cat([h, x_decode_flat], dim=1)

        return output


# Usage example and testing
if __name__ == "__main__":
    # Test the models with different input shapes

    print("Testing DEKMDenseModel...")
    dense_model = DEKMDenseModel(input_shape=784, hidden_units=10)
    dummy_input = torch.randn(32, 784)  # batch_size=32, input_dim=784
    dense_output = dense_model(dummy_input)
    print(f"Dense model input shape: {dummy_input.shape}")
    print(f"Dense model output shape: {dense_output.shape}")
    print(f"Expected output shape: (32, {10 + 784})")  # hidden_units + input_shape
    print()

    print("Testing DEKMConvModel...")

    # Test MNIST-like input (28x28x1)
    print("MNIST-like input (28x28x1):")
    conv_model_mnist = DEKMConvModel(input_shape=(28, 28, 1), hidden_units=10)
    dummy_input_mnist = torch.randn(32, 1, 28, 28)  # PyTorch format: (N, C, H, W)
    conv_output_mnist = conv_model_mnist(dummy_input_mnist)
    print(f"Conv model input shape: {dummy_input_mnist.shape}")
    print(f"Conv model output shape: {conv_output_mnist.shape}")
    print(
        f"Expected output shape: (32, {10 + 28 * 28 * 1})"
    )  # hidden_units + flattened_image
    print()

    # Test USPS-like input (16x16x1)
    print("USPS-like input (16x16x1):")
    conv_model_usps = DEKMConvModel(input_shape=(16, 16, 1), hidden_units=10)
    dummy_input_usps = torch.randn(32, 1, 16, 16)
    conv_output_usps = conv_model_usps(dummy_input_usps)
    print(f"Conv model input shape: {dummy_input_usps.shape}")
    print(f"Conv model output shape: {conv_output_usps.shape}")
    print(f"Expected output shape: (32, {10 + 16 * 16 * 1})")
    print()

    # Test FRGC-like input (32x32x3)
    print("FRGC-like input (32x32x3):")
    conv_model_frgc = DEKMConvModel(input_shape=(32, 32, 3), hidden_units=10)
    dummy_input_frgc = torch.randn(32, 3, 32, 32)
    conv_output_frgc = conv_model_frgc(dummy_input_frgc)
    print(f"Conv model input shape: {dummy_input_frgc.shape}")
    print(f"Conv model output shape: {conv_output_frgc.shape}")
    print(f"Expected output shape: (32, {10 + 32 * 32 * 3})")
    print()

    # Test with TensorFlow-like input format (N, H, W, C)
    print("Testing with TensorFlow-like input format (N, H, W, C):")
    dummy_input_tf_format = torch.randn(32, 28, 28, 1)  # TensorFlow format
    conv_output_tf = conv_model_mnist(dummy_input_tf_format)
    print(f"TF-format input shape: {dummy_input_tf_format.shape}")
    print(f"Conv model output shape: {conv_output_tf.shape}")
    print("Model automatically handles TensorFlow format conversion!")

    print("\nAll tests completed successfully!")

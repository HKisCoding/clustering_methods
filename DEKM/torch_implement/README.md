# DEKM PyTorch Implementation

This directory contains a PyTorch implementation of the DEKM (Deep Embedded K-Means) algorithm, converted from the original TensorFlow implementation.

## Files

- `dekm.py` - Contains the neural network models (DEKMDenseModel and DEKMConvModel)
- `trainer.py` - Contains the training logic (DEKMDenseTrainer class)
- `example.py` - Simple usage example with synthetic data
- `requirements.txt` - Python dependencies

## Key Conversion Details

### From TensorFlow to PyTorch

#### 1. Model Architecture (`dekm.py`)

**TensorFlow Original:**
```python
# Dense layers with specific architecture
filters = [500, 500, 2000]
input = layers.Input(shape=input_shape)
x = input
for i in range(len(filters)):
    x = layers.Dense(filters[i], activation="relu", kernel_initializer="uniform")(x)
x = layers.Dense(hidden_units, kernel_initializer="uniform")(x)
h = x
```

**PyTorch Conversion:**
```python
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
```

#### 2. Weight Initialization

**TensorFlow:** Used `kernel_initializer="uniform"`
**PyTorch:** Custom initialization method:
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

#### 3. Loss Functions

**TensorFlow Original:**
```python
def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)
```

**PyTorch Conversion:**
```python
def pretrain_loss(self, y_pred, y_true):
    # Extract reconstruction part (skip hidden units part)
    y_pred_recon = y_pred[:, self.hidden_units:]
    return nn.MSELoss()(y_pred_recon, y_true)
```

#### 4. Training Loop

**TensorFlow Original:**
```python
with tf.GradientTape() as tape:
    tape.watch(model.trainable_variables)
    y_pred = model(x[idx])
    y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], V)
    loss_value = losses.mse(y_true, y_pred_cluster)
grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**PyTorch Conversion:**
```python
optimizer.zero_grad()
y_pred = self.model(x[idx])
y_pred_cluster = torch.matmul(y_pred[:, :self.hidden_units], V_tensor)
loss_value = nn.MSELoss()(y_pred_cluster, y_true)
loss_value.backward()
optimizer.step()
```

#### 5. Data Loading

**TensorFlow:** Used `tf.data.Dataset`
```python
ds_xx = (
    tf.data.Dataset.from_tensor_slices((x, x))
    .shuffle(8000)
    .batch(pretrain_batch_size)
)
```

**PyTorch:** Uses `DataLoader`
```python
dataset = TensorDataset(x, x)  # Input and target are the same for autoencoder
dataloader = DataLoader(dataset, batch_size=self.pretrain_batch_size, shuffle=True)
```

### 6. Device Management

**TensorFlow:** Automatic GPU usage
**PyTorch:** Explicit device management:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
x = x.to(device)
```

## Key Features Preserved

1. **Two-phase training:** Pretraining (autoencoder) + Main training (clustering)
2. **K-means integration:** Regular K-means clustering on learned representations
3. **Hungarian algorithm:** For assignment matching between iterations
4. **Eigenvalue decomposition:** For within-cluster scatter matrix analysis
5. **Mini-batch training:** Efficient batch processing

## Usage

### Basic Usage

```python
from trainer import DEKMDenseTrainer
import torch

# Create trainer
trainer = DEKMDenseTrainer(
    input_shape=784,      # Input dimension
    hidden_units=10,      # Embedding dimension
    n_clusters=10,        # Number of clusters
    device='cuda'         # or 'cpu'
)

# Pretrain the autoencoder
trainer.pretrain(x_data)

# Main DEKM training
assignments = trainer.train(x_data, y_labels)

# Predict on new data
predictions = trainer.predict_clusters(x_new)
```

### Running the Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python example.py
```

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0 (for K-means)
- SciPy >= 1.7.0 (for Hungarian algorithm)

## Differences from TensorFlow Version

1. **Error Handling:** Added comprehensive error checking for missing dependencies
2. **Device Management:** Explicit GPU/CPU handling
3. **Modular Design:** Separated model and trainer into different classes
4. **Memory Management:** Better handling of tensor device placement
5. **Flexibility:** Easy to modify hyperparameters and training procedures

## Performance Notes

- The PyTorch version should have similar performance to the TensorFlow version
- GPU acceleration is available when CUDA is detected
- Memory usage is optimized with proper tensor device management
- The algorithm complexity remains O(n*k*d) per iteration where n=samples, k=clusters, d=dimensions

## Training Tips

1. **Pretraining:** Essential for good clustering performance
2. **Batch Size:** Use larger batches (256-512) for stable training
3. **Update Interval:** K-means updates every 10 iterations work well
4. **Convergence:** Monitor assignment changes to detect convergence
5. **Hyperparameters:** Hidden units should be much smaller than input dimension

## Troubleshooting

- **CUDA errors:** Check GPU memory and use smaller batch sizes if needed
- **Convergence issues:** Increase pretraining epochs or reduce learning rate
- **Poor clustering:** Try different hidden_units or n_clusters values
- **Memory issues:** Reduce batch_size or use CPU instead of GPU

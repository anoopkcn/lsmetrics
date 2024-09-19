# NOTES

## ADDED FEATURES
- Use the [original convolutional model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) as a base model.
- With **RBFCalculator** calculator for edge attributes rather than the original gaussian distance calculator, I get the slightly improved results compared to the [latest published results](https://arxiv.org/abs/2312.00111)
- I also extednded the atom features by adding the following as features:
  - atomic_number
  - electronegativity
  - atomic_radius
  - ionization_energy
  - electron_affinity
  - valence_electrons
- In the original implementaion edge features are NOT learnable but I made an option to add edge features in addtion to node features learnable. This modification allows the model to learn an embedding for the edge features, which can potentially capture more complex relationships in the graph structure.
  - This modification adds one more hyperparameter to the model but it is optional.
- I implemented attention mechanism and isomorphism in the model, but both these improvements doesn't seem to improve the model performance.

## TODOS
1. Add global features: Include global features of the crystal structure, such as lattice parameters or symmetry information.

2. Use a multi-task learning approach: If you have multiple related properties to predict, consider modifying your model to predict them simultaneously.


## TRAINING TABLE
local dataset = `perovskites_halide.json` 1200 samples
In all cases a weighted loss fuction is used(`0.4*mse_loss+0.4*l1+0.2*huber_loss`)
*trainable features

| Model | Edge feature Calculator | Node Feature Calculator | MAE |
|:-------|:-----|:------| ----:|
| CSGCNN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.218 |
| CSGCNN | RBFCalculator | AtomCustomJSONInitializer*  | 0.231 |
| CSGCNN | GaussianDistanceCalculator | onehot_encode_atom*  | 0.230 |
| CSGCNN | GaussianDistanceCalculator | atom_to_bit_features*  | 0.281 |

local dataset = `perovskites_10k.json` 10,000 samples

| Model | Edge feature Calculator | Node Feature Calculator | MAE |
|:-------|:-----|:------| ----:|
| CSGCNN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.44 |
| CSGANN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.70 |
| CSGINN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.63 |

Other pooling methods like attention based Set2Set Pooling doesnt improve the model performance.
## Extra

When to use:
- Complex graph structures where simple averaging might lose important relational information.
- Tasks requiring a richer graph representation that captures global context.

8. Use data augmentation:
  Implement data augmentation techniques specific to crystal structures, such as random rotations or translations.

2. Implement residual connections:
    Add skip connections to allow information to flow more easily through the network.

```python
class ResidualBlock(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.conv(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn(x)
        return x + residual

# In the CSGCNN class:
self.res_blocks = nn.ModuleList()
for _ in range(num_layers):
    conv = GraphConv(hidden_channels, hidden_channels)
    bn = nn.BatchNorm1d(hidden_channels, track_running_stats=False)
    self.res_blocks.append(ResidualBlock(conv, bn))

# In the forward method:
for res_block in self.res_blocks:
    x = res_block(x, edge_index, edge_attr)
```

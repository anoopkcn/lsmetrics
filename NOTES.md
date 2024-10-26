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

## TRAINING TABLE
local dataset = `perovskites_halide.json` 1200 samples
In all cases a MAE fuction is used as the loss function.
*trainable features

| Model | Edge feature Calculator | Node Feature Calculator | MAE |
|:-------|:-----|:------| ----:|
| **CSGCNN** | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.22 |
| CSGCNN | RBFCalculator | AtomCustomJSONInitializer*  | 0.22 |
| CSGCNN | GaussianDistanceCalculator | onehot_encode_atom*  | 0.27 |
| CSGCNN | GaussianDistanceCalculator | atom_to_bit_features*  | 0.28 |
| CSGCNN | WeightedGaussianDistanceCalculator | AtomCustomJSONInitializer*  | 0.26 |
| CSGCNN | AtomSpecificGaussianCalculator | AtomCustomJSONInitializer*  | 0.19 |
| FlowGNN | GaussianDistanceCalculator | AtomCustomJSONInitializer*  | 0.28 |
| FlowGNN | PeriodicWeightedGaussianCalculator | AtomCustomJSONInitializer*  | 0.30 |
| FlowGNN | AtomSpecificGaussianCalculator | AtomCustomJSONInitializer*  | 0.331 |


local dataset = `inorganic_SUNMAT_10k.json` 10,000 samples

| Model | Edge feature Calculator | Node Feature Calculator | MAE |
|:-------|:-----|:------| ----:|
| **CGCNN(ref)** | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.50 |
| **CSGCNN** | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.44 |
| CSGANN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.70 |
| CSGINN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.63 |
| CSGCNN | WeightedGaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.47 |
| FlowGNN | GaussianDistanceCalculator | AtomCustomJSONInitializer* | 0.495 |

Other pooling methods like attention based Set2Set Pooling doesnt improve the model performance.

## TODOS
- [ ] Add global features: Include global features of the crystal structure, such as lattice parameters or symmetry information.
- [ ] Use a multi-task learning approach: If you have multiple related properties to predict, consider modifying your model to predict them simultaneously.
- [ ] Gaussian regularization: Add a Gaussian regularization term to the loss function to encourage the model to learn smooth functions. Kullback-Leibler divergence can be used to measure the difference between the predicted distribution and a Gaussian distribution.


## Extra

Some recent studies have shown promising results for flow-based models in materials science:

- Köhler et al. (2022) introduced "Equivariant Flows," which outperformed previous methods in modeling atomic systems and predicting material properties.
- Shi et al. (2021) proposed a flow-based model for crystal structure generation that showed competitive performance with state-of-the-art methods.

However, transformer-based models have also shown strong performance:

- Xie et al. (2021) introduced "Crystal Transformer" for crystal property prediction, showing competitive results with graph neural networks.

## Contrastive Learning Metrics:
```python
def calculate_alignment_uniformity(encoder, dataloader):
    alignments = []
    uniformity = []

    for anchor, positive in dataloader:
        # Calculate alignment (similarity between positive pairs)
        anchor_enc = encoder(anchor)
        positive_enc = encoder(positive)
        alignment = F.cosine_similarity(anchor_enc, positive_enc).mean()
        alignments.append(alignment.item())

        # Calculate uniformity (distribution of features)
        uniformity.append(torch.pdist(anchor_enc).mean().item())

    return {
        'alignment': np.mean(alignments),
        'uniformity': np.mean(uniformity)
    }
```

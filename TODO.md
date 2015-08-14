#Data preprocessing
- dwi values are int16, so it should be normalized, either by:
  - using the b0 <-
    - do we just divide by the b0 (element-wise)?
  - using the int16.max()

- if using original hcp data, we should perform a top-up

- streamlines
  - resample streamlines so they have 100 points each.


#LSTM - regression

## Inputs
Diffusion weights in all available direction

## Outputs
1. Direction to follow
  - as-is
  - normalized
2. Continue or stop (using the binary cross-entropy)
  - Since it will be higly unbalanced, when the target is "stop" multiply the cost by the number of "continue" for a given streamline.


#LSTM - classification

## Inputs
Diffusion weights in all available direction

## Outputs
1. Softmax of the direction (something about softmax of gaussians)
2. Continue or not (using the binary cross-entropy)
  - Since it will be higly unbalanced, when the target is "stop" multiply the cost by the number of "continue" for a given streamline.
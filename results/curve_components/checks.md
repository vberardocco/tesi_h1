# Checks

1. k = 175 (all non-null PCs; there are 176 channels, one is silent in the calib): top 0.448332, bottom 0.448332, |difference| 0.0e+00. α chosen: top 31622.8, bottom 31622.8.
2. k = 175 against ridge_h50 (0.448332): 0.448332, difference +1.2e-07; equivalent weights on the channels against models/ridge_h50.npz: max diff. W 1.9e-17, b 1.2e-19; batch 1 0.448332. 80/20 validation R² 0.3922 (05_ridge_tune_long.py: 0.3922).
3. FA K = 15 refitted with the recipe: 0.100056; frozen model models/fa15_ridge_h50.npz evaluated now: 0.100056 (table B: 0.100056); max diff. refitted - frozen: npz 5.3e-15, loadings 5.7e-15.
4. Top curve, increments between consecutive k (5→10, 10→15, 15→25, 25→50, 50→100, 100→160, 160→175): +0.0527, +0.0497, +0.1193, +0.0403, +0.0539, +0.0653, +0.0337; monotonically increasing.
5. Every model: streaming batch 4 = block to the 4th digit; max diff. of the predictions 7.1e-10 (all models).
6. α at the edge of the grid: PCA_top k=15 (α 1), PCA_top k=25 (α 1).

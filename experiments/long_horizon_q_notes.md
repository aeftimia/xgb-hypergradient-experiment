# Long-horizon Q pilot

This branch tests an online value-function controller for XGBoost shrinkage.

At round `t`, the controller builds a compact training-state vector and considers candidate `eta` values. A small MLP predicts the change in validation log-loss `K` boosting rounds in the future. Labels are delayed: a decision made at round `t` only enters the replay set at round `t + K`, so the controller does not train on unavailable future information.

The first implementation searches a one-dimensional eta grid rather than differentiating through Q. In one dimension this is cheap and separates the value-model question from surrogate-gradient pathologies.

## Seven-seed pilot

| method | mean test log-loss | sd | mean AUC |
|---|---:|---:|---:|
| fixed tuned eta | 0.11047 | 0.04133 | 0.99120 |
| Q, K=5 | 0.11426 | 0.04631 | 0.99068 |
| Q, K=10 | 0.11473 | 0.04876 | 0.98994 |
| Q, K=20 | 0.11569 | 0.04626 | 0.99003 |

The learned controller does not beat the tuned fixed-eta baseline in this pilot. K=10 looked slightly better after the first three splits, but the advantage vanished after expanding to seven splits. This is useful evidence against the naive version rather than evidence against long-horizon control in general.

## Observed failure modes

1. A naive differentiable-Q version driven directly by gradient descent on the learned critic rapidly pushed eta to the edge of the allowed range. This is classic surrogate exploitation / extrapolation error.
2. Warmup exploration and a bounded eta grid make the controller much more stable, but the critic still has weak counterfactual coverage: each real trajectory observes only one action at each state.
3. The Wisconsin breast-cancer dataset is small enough that validation-set overfitting is a serious concern. Even explicit K-step validation rollouts can improve the validation objective while worsening held-out test loss.

## Next experiment

The most informative next step is to improve counterfactual supervision rather than make Q larger: periodically fork the current booster, try several eta values for K rounds, keep one branch for training, and use every fork as supervised data for Q. This tests whether the bottleneck is value-model capacity or lack of action coverage. A larger public tabular dataset should also replace Wisconsin before drawing conclusions.

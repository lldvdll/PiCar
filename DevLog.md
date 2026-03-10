Done
- Create baseline training code
- Base model
    - MobileNetV2 because it's compact and much faster both to train and to run on the Pi
    - Transfer learning, global average pooling to reduce CNN feature maps down to scalar activations to reduce head size
    - GAP (64 features, relu)
    - 2 dense (1, sigmoid) - for speed and angle, normalised to 0-1
- Set up Weights and Biases. Preferable solution for live logging and easy result sharing. Also far easier to set up than MLFlow
- Note: Model converges after epoch 3. Most likely because we freeze the CNN layers, the MLP are small so converge fast
- Run model and get baseline submission - plateus before epoch 5, the bottleneck is the frozen CNN layers
- Make head wider and deeper - no improvement, the bottleneck is the frozen CNN layers
- [REPORT] Unfroze 20 layers, with head warmup of 5 epochs. Interesting spike in loss when unfreezing, then not enough epochs for fine tuning to converge. Spike is "optimiser shock", effectively recompiling the model wipes momentums/history from Adam, so we take a suboptimal step
- Rerunning with 7 epochs for frozen and 25 epochs for unfrozen - not really much improvement
- De-biased training data, created new label file with weightings based on speed/angle joint distribution
- Re-running with 7 warm up epochs and 20 unfrozen epochs (20 layers)




Next
- Switch to binary speed. Also create  submission only output which snaps predictions to bins
- Note outputs are continuous - we map them to the discrete values expected for submission only
    - speed - 0,1
    - angle - 16? discrete bins?
- Implement gradual unfreezing - with appropriate learning rates

- Analyse bias in training data. If significant, balance it in training
- Extend head - make it both wider and deeper. Pay attention to epoch convergence
- Review relevant model list for more efficient
- Prune redundant features?
- Assess various pretrained models for task relevance, efficiency and performance

Done
- Create baseline training code
- Base model
    - MobileNetV2 because it's compact and much faster both to train and to run on the Pi
    - Transfer learning, global average pooling to reduce CNN feature maps down to scalar activations to reduce head size
    - GAP (64 features, relu)
    - 2 dense (1, sigmoid) - for speed and angle, normalised to 0-1
- Set up Weights and Biases. Preferable solution for live logging and easy result sharing. Also far easier to set up than MLFlow
- Note: Model converges after epoch 3. Most likely because we freeze the CNN layers, the MLP are small so converge fast
- [01] Run model and get baseline submission - plateus before epoch 5, the bottleneck is the frozen CNN layers
- [02] Make head wider and deeper - no improvement, the bottleneck is the frozen CNN layers
- [03][REPORT] Unfroze 20 layers, with head warmup of 5 epochs. Interesting spike in loss when unfreezing, then not enough epochs for fine tuning to converge. Spike is "optimiser shock", effectively recompiling the model wipes momentums/history from Adam, so we take a suboptimal step
- Rerunning with 7 epochs for frozen and 25 epochs for unfrozen - not really much improvement
- De-biased training data, created new label file with weightings based on speed/angle joint distribution
- [04] Re-running with 7 warm up epochs and 20 unfrozen epochs (20 layers) - improved performance and more epochs required (35)
- Create visualisation app for pre-processing - Crop can be more agressive 15 > 120 TOP, 0 > 30 BOTTOM, resize (320, 240) > (96, 160) [Note: any size not in 96, 120, 160, 224 defaults to weight size 224]. Crop is extreme and removes roadsigns. Can always relax later.
- [05] run 35 epochs. imroved results, but only slightly
- tested snapping. made mse worse, mainly because speed snapping is dramatic (0,1). Will test each training run
- [06] implemented data augmentation - colour, light, noise, vertical tilt, and rotation only. Might add more later
- Higher MSE, likely due to too much rotation/tilt affecting road marking to angle mapping as angle loss is now higher.
- [07] Added attention block, also Conv bottleneck before, and 2 layer MLP after. Reduce tilt and rotation as they may confuse angles





Next
- Try mobilenetv3, smaller.
- Add conv layer to head pre MLP ()
- Add transformer layer to head pre MLP (preserve spatial context)
- Switch to binary speed. Also create  submission only output which snaps predictions to bins
- Note outputs are continuous - we map them to the discrete values expected for submission only
    - speed - 0,1
    - angle - 16? discrete bins?
- Implement gradual unfreezing - with appropriate learning rates
- clean up glare on road mat
- remove as many dead or useless images as i find in pre-processing analysis
- Analyse bias in training data. If significant, balance it in training
- Extend head - make it both wider and deeper. Pay attention to epoch convergence
- Review relevant model list for more efficient
- Prune redundant features?
- Assess various pretrained models for task relevance, efficiency and performance
- FINAL PASS: Train blind on ALL 14k images. Train with validation first to find epoch to stop at. Then train on ALL data, no validation. Apparently this always does better in kaggle because the model gets to see more data.

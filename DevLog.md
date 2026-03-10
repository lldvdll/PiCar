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




Next
- Unfreeze some CNN layers - run frozen for 5 epochs to warm up the head, then finetune the MobileNetV2 model
- Note outputs are continuous - we map them to the discrete values expected for submission only
    - speed - 0,1
    - angle - 16? discrete bins?
- Analyse bias in training data. If significant, balance it in training
- Extend head - make it both wider and deeper. Pay attention to epoch convergence
- Unfreeze some CNN layers. Let the features learn task specifics. Semantic features only


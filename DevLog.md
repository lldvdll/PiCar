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
- Rand for 35 epochs. It did worse again. Geometric data augmentation is to agressive and will affect angle prediction. Last MobileNetV2 layer is only 3x5, so maybe actually need to chop some layers off MobileNet for transformer to have space to work? Or just drop it
- Run feature analysis - Grad-CAM for speed and angle independent, and feature maps over layers. Notice network is still learning background cheats. Some clearly following road markings, though inconsistent flip-flip for speed and angle between images. Sometimes not caring enough about both lines.Some clear obstruction detection. Block 7 and beyond features look a bit useless. Cutout augmentation will help with overfitting to specific line arangements. Drop blocks 10>, they are a bit useless and give the attention block less space, but unfreeze 7-10 for learning specific features before attention. ROI pre-processing, e.g. polygon masking, to recover road signs and block out distracting background at image corners.
- [08] Next - fix augmentation, unfreeze more layers (40), rerun 
- Check for other weights for mobilenetv2! imagenet might be a bad starting point. Actually the only pretrained option was imagenet...
- [09] Run planned reduced augmentation (reduce rotation, tilt, and some more bad image removal, blind unfreeze 40 layers). Running now. No improvement, angles worse, weird validation loss spikes at epoch 36 and 46.
- [09] Image loading was taking 2 mins per epoch! Now caching resized images in epoch 1, then still applying augmentation each epoch. 1 hour quicker for 30 epochs!
- [10] Added cutout augmentation, randomly mask the training images with black rectangles as a form of regularisation so the model can more consistently learn to map sets of features rather than focusing on specific features.
- [11] Looks much better. Need more epochs to converge, best val was 2nd last epoch. Will run again on GPU.
- [12] Huge step in improvement, angle still not converged after 100 epochs. Speed looks like it might be converging. Next run for 200 epochs
- [13] 200 still converging though leveling out. Trying 500 to be certain.
- [14] Note: Been freezing by layer number so far, cutting blocks in half! Need to adjust the code to use layer name instead so i can always ensure cutting/freezing non-destructively at block boundaries.
- Run with correct block based freezing. Slight improvement
- [15] Now running with cuts - cut at block 10, freeze at block 6. In theory the deeper layers have learned too many specifics which are too detailed for our needs. We keep the trunk of the network, chop off the unnecesary details, and let the attention heads resolve the global spatial relationships. Marginally worse. Way faster convergence
- [16] Next add an extra attention block. Not really any improvement at all.
- [17] Split the attention blocks now. Slight improvements, but still not hugely better
- [18] Less agressive alround - layer cut 10>12, cutout prob 05.>0.2, cutout max_pixels 50>30, crop 110,30>100,20
- [19] Next experiment i should just unfreeze it all, set a graded learning rate, stop chopping the network, and remove the transformer blocks, and see how this does. Unfreeze all, but one block at a time and with decreasing learning rate. Also, removed the cut. Much smoother and more consistent learning curve, though speed flattens out early. But somehow much worse in kaggle
- [20] Try without attention now. Validation now deviating from train a lot more in fewer iterations. Maybe the learning rate is too high to start with or something?
- [21] Fewer epochs per layer, lower base learning rate, fix validation issue, use huber loss, use sqrt of weights for balancing, implement horizontal flip augmentation, remove bottom crop
- Angle is learning really well, but speed is way worse. Infact it's the validation loss in speed tht's way higher than train and brings the whole score down.
- [22] Remove sqrt from the debias process. Speed 0 is being undersamples and not learned correctly
- Might try keeping a few layers frozen. Then also increase number of epochs per layer.
- Speed loss is bad. I need to figure out what the model 13 did right. Otherwise just create a frankenstein output which combines the best of speed and angle? 
- Need complete analysis of speed errors - add speed/turning options to compreensive analysis. Also regenerate the joint distribution plot but plot average error
- So much bad data! Removed hundreds of junk files and made hundreds of speed and angle corrections. I notice the current model understeers on bends. Then realised the bad data is also in the test dataset!
- [23] 20 epochs per layer, keep last 6 frozen. Also added some things like prediction file generation at the end.
- [24] use larger images. also crop the bottom out again, the wheels don't always match the angle. Definite improvement, but not significant
- [25] efficientnetb3. Did better. Still not great with speed. Trying higher learning rate - 1e-4 with 0.9 decay


- Consider changing speed to binary output and relevance of huber loss. Also consider snapping/binning angle


- After i should try fix validation, use huber loss, and add flip augmentation.
- Then i should make everything more agressive again?
- Actually i notice the current models do worse on speed. Most likely kaggle score is worse because speed gets a huge penalty if incorrect. So need to step back to that state if possible. Looks like the switch may have been the introduction of attention? Yep, looks like attention was breaking the speed scores.
- Are wheels a big angle predictor? Should we add them back in for kaggle and then remove them again for the live car?
- I should also snap the angles, even if i leave the speed as is.
- Also, need to recalculate the sample weights. And probably set the balancing to sqrt of the weight because those rare images will be getting heavily oversampled. May also consider only balancing the speed 1 data, since speed 0 is sparse?
- need to encorporate label corrections into data pipeline. I see some spurious values
- important arrows are definitely getting cropped.


Next
- cut after block 10 - i think for this i would like to cut blocks 10-17, but freeze blocks 1-6, giving the model space to learn new features, but deleting the deeper stuff in place of the current transformer block. But please you'll need to help me identify the correct layers, or add a layer to block mapping to the code.
- ROI masking - this would allow us to be less agressive in the centre,
 recovering important road signage, while also cutting useless background from the outer edges.
-Splitting the transformer layers - again, makes perfect sense to decouple the context analyser component for the different objectives. It also makes a bit of sense to combine them because they are not independant, but let's try this first.
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

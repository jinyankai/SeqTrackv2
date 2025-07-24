## front_discriminator
### function:
### 1.1 
- use the predicted information of the kf_tracker to predict the next bbox
- use the predicted and expanded bbox to roi the region of next picture (assert bigger than the smallest threshold)
- send the roi region and the class name into the CLIP score to estimate whether the object is missing
### 1.2 
contain:
- clip_scorer : read the roi and the text information. give out the pairing score 
- predictor : class Predictor is used to manage the bbox given by the seqtrackv2 and update the KF_tracker, and gives the predicted and expanded bbox, then roi the next picture and prepare the data for clip scorer


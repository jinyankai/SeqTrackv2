## KF_tracker
### function:
#### 1.1 
based on kalman-filter, to track the center xy and depth_z of tracked object, contains update and predict methods

file1 :
- **class kf_tracker** :
    - update method : take a xyz list as input and use it to update tracker
    - predict method : predict new time step xyz
    - predict distribution: give out the most likely area of the predict object on image (or u can say the xy)

file2:
- class kf_tracker_manager :
  - manage the tracker
  - initialize method : used to initialize the kf_tracker, take the initial xyz
  - kill method : take a bool value which signals the kill message, kill the tracker when we found that the tracked object is missing
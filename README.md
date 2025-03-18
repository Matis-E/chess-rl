# Chess - RL

 - Here's a readme explaining how our project is structured and in which order to run the test codes.

 # Training of models

 - dqn.py corresponds to the basic training file for the dqn model. 
 - dqn2015.py corresponds to the dqn2015 model seen in tp. 
 - Then CEM.py and (1+1)_SAES correspond to the training files for the CEM and (1+1)_SAES models. 
 
 To train the models, simply run the python codes. To save the models, specify where they are to be saved. We have saved the models in pth format in the corresponding folders.

# Testing models

 - dummy vs random.py test games between the model that takes the best reward on each move and the model that takes a random move
 - dummy vs dqn.py test games between dummy and the dqn model (dnq or dnq 2015)
 - dummy vs cem.py test games between dummy and the cem model 
 - and dqn vs cem.py test games between dqn and  cem  
 - model_versus_player.py allow player to play against cem model but the shot format is a bit unusual. It corresponds to the piece's starting position and its finishing position. The axis starts at the top left of the chessboard and takes into account first the abscissa and then the ordinate. 

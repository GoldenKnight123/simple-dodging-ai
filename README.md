# simple-dodging-ai
 A very simple AI placed in an environment with random bullet hell spam. Built using a NN and DQN agent.

# States Format:
 [(At top edge (0: False, 1: True)), <br/>
(At bottom edge (0: False, 1: True)), <br/>
(At left edge (0: False, 1: True)), <br/>
(At top edge (0: False, 1: True)), <br/>
(Action of the agent (0: Left, 1: Right, 2: Up, 3: Down, 4: Up Left, 5: Up Right, 6: Down Left, 7: Down Right, 8: Stay Still)), <br/>
(Distance to closest bullet within 64 pixels from center of player (if any)), <br/>
(Angle of the closest bullet within 64 pixels from center of player from the player (if any)), <br/>
(Angle of movement of the closest bullet within 64 pixels from center of player (if any)), <br/>
(Distance to second closest bullet within 64 pixels from center of player (if any)), <br/>
(Angle of the second closest bullet within 64 pixels from center of player from the player (if any)), <br/>
(Angle of movement of the second closest bullet within 64 pixels from center of player (if any)), <br/>
(Distance to third closest bullet within 64 pixels from center of player (if any)), <br/>
(Angle of the third closest bullet within 64 pixels from center of player from the player (if any)), <br/>
(Angle of movement of the third closest bullet within 64 pixels from center of player (if any))]
<br/><br/>
# Model Architecture:
Input(Flatten Layer): 14 <br/>
Hidden Layer 1(Dense Layer): 16 <br/>
Output (Dense Layer with Softmax): 9 <br/>
<br/><br/>
# Highest Score Reached by Pretrained Model: 607

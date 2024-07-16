# VisionTag

This is the code for a laser tag like prototype which uses AI vision for hit registration. 
![VisionTagAnnotated](https://github.com/user-attachments/assets/6c770c54-26a1-4179-a318-f5274d852153)

Here is a [link to the youtube video](https://youtu.be/RA7jHUNauQk) which gives a good overview. (Headshots were not in place for this video during the playtesting segment)

### The devices and inferencing server can:
- See players from 50+ feet away even if only their head is showing thanks to the 10x zoom camera
- Register headshots for extra damage
- Run in 2v2 mode with colored headbands and shirts annotating player 1 and player 2 on each team
- Hit data can come back in 1/8th of a second
- Health and gun parameters (damage, fire rate, fire mode, magazine size) can all be configured server side on the fly

  ## How it works
When the trigger is clicked on a device the raspberry pi reads the trigger and begins sending the concatenation of the latest images from the 2 cameras, one 10x zoomed camera and one regular camera. Once the server recieves the concatenated image it runs a segmentation inference on it from a model I fine tuned. The model is based on the yolov8-segmentation-nano model and the fine tuning was to have a head annotation as well as body so that headshots could be detected with fast inference times on my laptop.

Once the inference is done the center of the two frames is checked for being inside a body or head segmentation mask, if the center is part of the mask then the damage for the segmentation class and device is sent out to the opposing device. If 2v2 is enabled it runs another AI vision model custom trained to detect if the player is wearing a orange headband or not, this classification determines which of the 2 players the damage goes to.

The shooting device will make a sound that notifying that they hit the opposing player. The player hit will have the handle of their device vibrate as well as their health light dimming or changing color. Once killed, a device will play the mario death sound effect and will take a few seconds to become active again while the opposing player's device will play a sound denoting the kill. There is also a reload button that takes a few seconds and prevents the player from shooting in the meantime.

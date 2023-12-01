# IOT_Face_Recognition
The problem statement for the flow chart is:

=> Design a system that can recognize faces and store them in a database for future comparison.

=> The system should be able to take a photo of a person using a camera, process the image to extract the face features, and store the image in a database with a unique identifier.

=> The system should also be able to fetch an image from the database and compare it to a new image taken by the camera, and perform an operation based on the result of the comparison.

=> The system should use a face recognition algorithm to determine the similarity between two images and output a score or a label.

What have we achieved?
30-11-2023 => Through the pre-trained haarcascade file we are able to detect face and eyes in the face
01-12-2023 => Through the imwrite function saving the frame with the detected face
Task : @Get a cropped image of the face only : Done
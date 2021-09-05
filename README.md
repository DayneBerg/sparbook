# sparbook
Image-to-text transcription application powered by ML and a unique segmentation algorithm.

This project was started around the start of the fall 2020 semester and became a github repo around the start of the spring 2021 semester. Development and documentation is ongoing, but dependent on my free time.

The app is designed for use with images of book pages. The user first selects a paragraph in their image, then that paragraph is segmented into multiple lines and transcribed by a neural network. The user then corrects any transcription errors and appends the new block of text to their queued text. This queued text may be saved to a directory at any time. The user may also choose to collect transcription examples as they work to later be contributed to this project's aspirational image-to-text dataset. 

The project is currently in a functional state, and I would be happy to receive feedback on it (particularly on how to make it more user-friendly). 

Future work includes:
-Improving the neural network
-Packaging the application in a more familiar way
-The ability to locally train a network behind-the-scenes
-Support for specfic file types when saving

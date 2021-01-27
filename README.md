# sparbook
Image-to-text transcription application powered by A.I. and a unique segmentation algorithm.

This project was started around the start of the fall 2020 semester and became a github repo around the start of the spring 2021 semester. Development and documentation is ongoing, but dependent on my free time.

The app is designed for use with images of book pages. The user first selects a paragraph in their image, then that paragraph is segmented into multiple lines and transcribed by a neural network. The user then corrects any transcription errors and appends the new block of text to their queued text. This queued text may be saved to a directory at any time. The user may also choose to collect transcription examples as they work to later be contributed to this project's aspirational image-to-text dataset. 

Currently the interface, segmentation algorithm, and database-building functionality are essentially complete. Future work includes finalizing the neural network (which is currently in dire need of debugging) and various miscellaneous tasks (evident in my many TODO statments).

ML model for Heart Disease Detection
===========================

Trained a heart disease detection model using ensemble learning method. Used KNN, SVM, Decision Tree and Random Forest classifier outputs to train the final Decision Tree classifier.


1. Build docker image
---------------------
.. code-block::

    docker build -t heart .

2. Run container
----------------
.. code-block::

    docker run --name ml_api -p 8000:8000 heart

3. Output will contain
----------------------
INFO:     Uvicorn running on http://0.0.0.0:8000

Use this url in chrome to see the model frontend;
use http://0.0.0.0:8000/docs for testing the model in the web interface.

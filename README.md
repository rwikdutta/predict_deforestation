# predict_deforestation
This project was done for NASA India Space Apps Hackathon Prelims where we used data from ( https://kaggle.com/c/planet-understanding-the-amazon-from-space ) challenge to train a model and then use images from NASA's Landsat 8 satellite via their API to predict deforestation related activities and serve it as an API 

The Forests.ipynb contains the training code that was used.

The nasa_model_server_django folder contains the django project which serves up the model as a REST Web service. Please note that the code written for the REST Web service was just written for demo purposes and has not been modified since so there might be quite a few quirks there.

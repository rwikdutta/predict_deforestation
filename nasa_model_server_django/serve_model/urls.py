from .views import ServeDeforestationModel
from django.conf.urls import url

urlpatterns=[
    url(r'^predict_deforestation/$',ServeDeforestationModel.as_view()),
]
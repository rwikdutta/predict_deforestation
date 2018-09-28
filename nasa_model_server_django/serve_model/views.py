from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView, status
from datetime import date
from NASA_Hack_Prelims_2.constants import *
import requests
import os
import time
import numpy as np
import NASA_Hack_Prelims_2.urls
from NASA_Hack_Prelims_2.urls import *


class ServeDeforestationModel(APIView):

    def get(self, request):
        if request.GET.get('latitude'):
            latitude=request.GET.get('latitude')
        else:
            return Response({'error': True, 'Message': 'No Latitude Given'}, status=status.HTTP_200_OK)
        if request.GET.get('longitude'):
            longitude = request.GET.get('longitude')
        else:
            return Response({'error': True, 'Message': 'No Longitude Given'}, status=status.HTTP_200_OK)
        cloud_score=False
        #token=API_KEY
        date_today=date.today()
        url='https://api.nasa.gov/planetary/earth/imagery/?lon={}&lat={}&date={}&cloud_score={}&api_key={}'.format(longitude,latitude,str('2015-06-06'),str(cloud_score),API_KEY)
        print(url)
        r=requests.get(url)
        if r.status_code==200:
            resp=r.json()
            pic_url=resp['url']
            re=requests.get(pic_url,allow_redirects=True)
            if re.status_code==200:
                file_name='{}_{}.png'.format(int(time.time()),np.random.randint(50000))
                open(file_name,'wb').write(re.content)
            else:
                return Response({'error': True, 'Message': 'Problem retrieving picture from NASA'}, status=status.HTTP_200_OK)

        else:
            return Response({'error': True, 'Message': 'Problem accessing NASAs API'}, status=status.HTTP_200_OK)
        cwd=os.getcwd()
        preds=NASA_Hack_Prelims_2.urls.predict('{}/{}'.format(cwd,file_name),size=256)
        preds_2=NASA_Hack_Prelims_2.urls.weighted_predict(preds)
        pred_dict={}
        for percent,cat in preds_2:
            pred_dict[cat]=percent
        return Response({'error': False, 'Message': 'On way to success','Preds':pred_dict}, status=status.HTTP_200_OK)



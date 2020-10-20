#!/usr/bin/env python
# @Time    : 2020/9/22 10:35
# @Author  : wb
# @File    : urls.py

# 配置URL

from django.urls import path

from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    path('map', views.show_map, name='show_map'),
    path('realtime', views.realtime_statistics, name='realtime_statistics'),
    path('incremental', views.show_incremental, name='show_incremental'),
    path('historical_list', views.historical_list, name='historical_list'),
    path('historical_data/<int:his_id>', views.historical_data, name='historical_data'),
    path('map_position', views.map_position, name='map_position'),
]





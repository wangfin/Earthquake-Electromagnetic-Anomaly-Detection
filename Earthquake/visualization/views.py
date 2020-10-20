from django.shortcuts import render

# Create your views here.


def index(request):
    return render(request, 'index.html')

def show_map(request):
    # 展示地图热力图
    return render(request, 'map.html')

def realtime_statistics(request):
    # 展示实时数据展示
    return render(request, 'realtime_statistics.html')

def show_incremental(request):
    # 展示增量对比能力
    return render(request, 'incremental_learning.html')

def historical_list(request):
    # 历史数据列表
    return render(request, 'historical_list.html')

def historical_data(request, his_id):
    # 历史数据展示
    return render(request, 'historical_data.html')

def map_position(request):
    if request.method == 'GET':
        # 采用的是GET传递数据的方式，其实可以只传递ID，然后用ID去数据库中查询
        # 经度
        longitude = request.GET.get('lon', default='109.755121')
        # 纬度
        latitude = request.GET.get('lat', default='18.405743')

        coordinate = {'lon': longitude, 'lat': latitude}

    return render(request, 'map_position.html', {'coordinate': coordinate})


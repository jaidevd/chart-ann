from base64 import urlsafe_b64decode
from gramex.data import filter as gfilter
from gramex.config import variables


def view(handler):
    handler.set_header('Content-Type', 'image/png')
    handler.set_header('Content-Disposition', 'attachment; filename=image.png')
    data = gfilter(variables['DB_URL'], table='charts', args={'chart_id': [handler.path_args[0]]})
    url = data.iloc[0]['image'].split(',')[1]
    data = urlsafe_b64decode(url)
    return data

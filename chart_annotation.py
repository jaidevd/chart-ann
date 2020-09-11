from base64 import urlsafe_b64decode


def view(data, path, handler):
    if handler.request.method == 'GET':
        handler.set_header('Content-Type', 'image/png')
        handler.set_header('Content-Disposition', 'attachment; filename=image.png')
        url = data.iloc[0]['image'].split(',')[1]
        data = urlsafe_b64decode(url)
        return data

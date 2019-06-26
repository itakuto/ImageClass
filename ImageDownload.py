from google_images_download import google_images_download

res = google_images_download.googleimagesdownload()
arguments = {"keywords":"リンゴ",
             "limit": 100,
             "format": "jpg"
             }

res.download(arguments)

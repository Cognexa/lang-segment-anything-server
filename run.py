import requests
import json

response = requests.post(
    "http://localhost:8082/segment/",
    files=[
        ('photos', open('assets/DJI_20240729120840_0001_T_point0.JPG', 'rb').read()),
        ('photos', open('assets/DJI_20240729120840_0001_T_point0.JPG', 'rb').read()),
        ('photos', open('assets/DJI_20240729120840_0001_T_point0.JPG', 'rb').read()),
        ('photos', open('assets/DJI_20240729120840_0001_T_point0.JPG', 'rb').read())
    ],
    data={'parameters': json.dumps({
        "box_threshold": 0.15,
        "text_threshold": 0.25,
        "text_prompt": "square",
        "min_width": 20,
        "max_width": 80,
        "min_height": 10,
        "max_height": 60,
        "avg_color_darkness_limit": 50.0
    })}
)
print(json.dumps(response.json(), indent=2))

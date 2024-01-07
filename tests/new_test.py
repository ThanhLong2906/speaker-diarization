import os.path
import unittest
import httpx
import json
import time

class Test_DiarizationModule(unittest.TestCase):
    client = httpx.Client()
    host = 'http://172.16.1.20:8000'

    def test_add_voice(self):
        with open("/home/long/audio-service/voice_db/Le_Hoang.wav", 'rb') as file:
            files = file.read()
        response = self.client.post(f"{self.host}/add_voice",
                         files = files,
                         params = {'partition_name': 'test'},
                         timeout = 200)
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        response = self.client.delete(self.host + '/deletePartition', params={'partition_name': 'hehe'})
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    # init_app()
    unittest.main()
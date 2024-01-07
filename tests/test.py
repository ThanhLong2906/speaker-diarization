import os.path
import unittest
import httpx
import json
import time

class Test_DataModule(unittest.TestCase):
    client = httpx.Client()
    host = 'http://localhost:5005'
    host_prompt = 'http://localhost:8000'

    def test_upload_docs_and_header_split(self):
        self.client.post(f'{self.host}/config/hehe', json={
            "chunk_size": 2000
            })
        with open('/home/hung/Downloads/SỔ TAY TTNC update 30-11-2016.docx', 'rb') as file:
            files = {'files': ('vtc.docx', file.read())}
        response = self.client.post(self.host + '/uploadDocsSync/',
                                    files=files,
                                    params={'partition_name': 'hehe'},
                                    timeout=200
                                    )
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        response = self.client.delete(self.host + '/deletePartition', params={'partition_name': 'hehe'})
        self.assertEqual(response.status_code, 200)

    def test_upload_duplicate(self):
        with open('/home/hung/Downloads/Bo TTTT.docx', 'rb') as file:
            files = {'files': ('bottt.docx', file.read())}
        response = self.client.post(self.host + '/uploadDocsSync/',
                                    files=files,
                                    params={'partition_name': 'huhu'},
                                    timeout=200
                                    )
        self.assertEqual(response.status_code, 200)
        response = self.client.post(self.host + '/uploadDocs/',
                                    files=files,
                                    params={'partition_name': 'huhu'},
                                    )
        self.assertEqual(response.status_code, 409)
        response = self.client.delete(self.host + '/deletePartition', params={'partition_name': 'huhu'})
        self.assertEqual(response.status_code, 200)

    def test_upload_no_ext(self):
        with open('/home/hung/Downloads/cam-nang-chuyen-doi-so.pdf', 'rb') as file:
            files = {'files': ('camnangso', file.read())}
        response = self.client.post(self.host + '/uploadDocs/',
                                    files=files,
                                    params={'partition_name': 'huhu'},
                                    )
        self.assertEqual(response.status_code, 409)

    def test_upload_multi_files(self):
        paths = ['/home/hung/Downloads/cam-nang-chuyen-doi-so.pdf', '/home/hung/Downloads/Về VTC.docx']
        file_list = []
        for path in paths:
            with open(path, 'rb') as file:
                files = ('files', (os.path.basename(path), file.read()))
                file_list.append(files)

        response = self.client.post(self.host + '/uploadDocs/',
                                    files=file_list,
                                    params={'partition_name': 'haha'},
                                    )
        self.assertEqual(response.status_code, 200)

    def test_delete_partition(self):
        response = self.client.delete(self.host + '/deletePartition', params={'partition_name': 'haha'})
        self.assertEqual(response.status_code, 200)
        response = self.client.delete(self.host + '/deletePartition', params={'partition_name': 'hehe'})
        self.assertEqual(response.status_code, 200)

    def test_delete_file(self):
        initial_sorry_msg = "Xin lỗi, tôi không tìm thấy văn bản nào cho câu truy vấn của bạn."
        paths = ["/home/hung/Downloads/tài liệu dịch vụ công/Dịch vụ công trực tuyến của UBND tỉnh Bạc Liêu.docx"]
        file_list = []
        for path in paths:
            with open(path, 'rb') as file:
                files = ('files', ('hehe.docx', file.read()))
                file_list.append(files)

        self.client.post(self.host + '/uploadDocs/',
                                    files=file_list,
                                    params={'partition_name': 'test'},
                                    )
        self.client.post(f'{self.host}/config/test4', json={
            "ooc_message": initial_sorry_msg
            })
        
        time.sleep(10)
        self.client.delete(self.host + '/deleteFile', params={'partition_name': 'test', 'file_name': 'hehe.docx'})

        response = self.client.post(self.host_prompt + "/NewPromptSync", json={
            "prompt": "thủ tục đăng ký kết hôn",
            "conversationId": "test",
            "partition_name": "test"
        }, timeout=200)
        print(response.json()['message'])
        self.assertEqual(response.json()['message'].startswith(initial_sorry_msg), True)

if __name__ == '__main__':
    # init_app()
    unittest.main()

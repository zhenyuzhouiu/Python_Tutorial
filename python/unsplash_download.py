import json
import multiprocessing
import datetime
import requests
import os
# BeautifulSoup可以把html文件或XML文件解析成树形结构
from multiprocessing import Pool
import urllib
import random
import cv2


def getHtml(url, headers):
    print('start-getHtml')
    # user-agent用于伪装浏览器的User Agent
    with requests.get(url, headers) as f:
        # data is bytes class
        # data = f.content
        # text is str class
        json_html = f.text
        print('Status:' , f.status_code, f.reason)
        # print('Headers:', f.headers)
        # json.load and json.loads
        dict_html = json.loads(json_html)
        return dict_html

def imgDownload(dict_html, num_page, target, folder_path, headers):
    # bf_html = BeautifulSoup(html, features="html.parser")
    # img_class = bf_html.select('div.img._2UpQX')
    x = 0
    # dict_html中方括号括住的是list，{}扩住的是dictionary
    for i in range(len(dict_html['results'])):
        save_path = folder_path + '{c}-{a}-{b}.jpg'.format(c=target, a=num_page, b=x)
        # 这样写可能会被网站组织了这类访问，需要在请求头上加上伪装成浏览器的header
        # urllib.request.urlretrieve(dict_html['results'][i]['urls']['raw'], save_path)
        html = requests.get(dict_html['results'][i]['urls']['raw'], headers)
        # 以byte形式将图片数据写入
        with open(save_path, 'wb') as file:
            file.write(html.content)
            file.flush()
        file.close()
        print("Downloading {c}-{a}th image, download link is {b}".format(c=num_page, a=x, b=dict_html['results'][i]['urls']['raw']))
        x += 1
    return 0

def iniHeader(target):
    Referer = 'https://unsplash.com/s/photos/' + target
    # User-Agent列表，用于伪装浏览器的User Agent
    USER_AGENTS = [
        "Mozilla/5.0 (Windows; U; Windows NT 5.2) Gecko/2008070208 Firefox/3.0.1"
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
        "Opera/9.80 (Windows NT 5.1; U; zh-cn) Presto/2.9.168 Version/11.50",
        "Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0",
        "Mozilla/5.0 (Windows NT 5.2) AppleWebKit/534.30 (KHTML, like Gecko) Chrome/12.0.742.122 Safari/534.30",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0",
        "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.2)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
        "Mozilla/4.0 (compatible; MSIE 5.0; Windows NT)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.2) Gecko/2008070208 Firefox/3.0.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070309 Firefox/2.0.0.3",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070803 Firefox/1.5.0.12 "
    ]
    # IP地址列表，用于设置IP代理
    IP_AGENTS = [
        "http://58.240.53.196:8080",
        "http://219.135.99.185:8088",
        "http://117.127.0.198:8080",
        "http://58.240.53.194:8080"
    ]
    # 设置IP代理
    proxies = {"http": random.choice(IP_AGENTS)}
    Cookie = 'ugid=55901a9e8cdccd18514cf3f4d576b3e85408694; lux_uid=162329072439354584; _sp_ses.0295=*; _sp_id.0295=f75301f3-4b95-4cee-a382-58684e315b0a.1623290724.1.1623290756.1623290724.49a2f47e-1595-4a3c-92fb-ad1bd09c869c; uuid=51886920-c990-11eb-a5a3-eb51aaebb6c3; xpos=%7B%22greater-uploader-limit%22%3A%7B%22id%22%3A%22greater-uploader-limit%22%2C%22variant%22%3A%22disabled%22%7D%7D; azk=51886920-c990-11eb-a5a3-eb51aaebb6c3; azk-ss=true; _ga=GA1.2.1500907768.1623290725; _gid=GA1.2.1246873963.1623290725; _gat=1'

    headers = {
        # 'User-agent': random.choice(USER_AGENTS),
        'User-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Cookie': Cookie,
        'Connection': 'keep-alive',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.5',
        'Host': 'unsplash.com',
        'Referer': Referer
    }
    return headers

def Project(url):
    # url = 'https://unsplash.com/napi/search/photos?query=' + target + '&per_page=20&page=' + str(num_page) + '&xp='
    # from the url get the target
    target = url[url.find('=')+1:url.find('&')]
    # from the url get the num_page
    num_page = url[url.rfind("page")+5:url.rfind('&')]
    # 设置requests请求的 headers
    headers = iniHeader(target)

    # requests get请求
    # html = getHtml(url, headers=headers, proxies=proxies)
    dict_html = getHtml(url, headers=headers)
    target = target.replace(" ", "-")
    # image save folder
    folder_path = './photo/' + target + '/'
    # 判断文件夹是否已经存在
    if os.path.exists(folder_path) == False:
        # 创建文件夹
        os.makedirs(folder_path)
    imgDownload(dict_html, num_page, target, folder_path, headers=headers)

def avgImgsize(path):
    if os.path.exists(path):
        num_img = 0
        sum_width = 0
        sum_height = 0
        for name in os.listdir(path):
            if name.split('.')[1] == 'jpg':
                img = cv2.imread(path + '/' + name)
                width, height, _ = img.shape
                sum_width = sum_width + width
                sum_height = sum_height + height
                num_img += 1
        print("The number of image: ", str(num_img))
        print("The average width of image is: ", str(float(sum_width/num_img)))
        print("The average height of image is: ", str(float(sum_height/num_img)))
    else:
        print("Please give a right image file path! The {a} path don't exists".format(a=path))


if __name__ == '__main__':
    target = input("Input keywords what kind of image you want to download:")
    num_pictures = int(input("How many pictures that you want to download:"))

    urls = []
    # 根据电脑CPU的内核数量创建相应的进程池
    pool = Pool(multiprocessing.cpu_count())
    for num_page in range(1, (num_pictures//20)+1):
        url = 'https://unsplash.com/napi/search/photos?query=' + target + '&per_page=20&page=' + str(num_page) + '&xp='
        # Project(url)
        urls.append(url)
    starttime = datetime.datetime.now()
    # 通过map方法去执行我们的主函数
    pool.map(Project, urls)
    # 让它不再创建进程
    pool.close()
    # 让进程池的进程执行完毕再结束
    pool.join()
    endtime = datetime.datetime.now()
    print("Download time is: " + str((endtime-starttime).seconds))

    # path = '/home/zhenyuzhou/Desktop/Yolo_mark/x64/Release/data/img'
    # avgImgsize(path)
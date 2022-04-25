import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os

# Chose out webpage
target = 'https://www.learnersdictionary.com/word-of-the-day'
# Requested our webpage
# res = requests.get(target, headers={"User-Agent":"Chrome/74.0.3729.169"})
res = urlopen(target)

# Requested text(HTML)
res.text
# print(res.text)
print(res.status_code)
# Created a beautiful soup object using the HTML / parsed
soup = BeautifulSoup(res.text, features="html.parser")
# Selected specific elements based off of element and class
word_html = soup.select('span.hw_txt')
print(word_html)
# Access the text content from the element
word_text = word_html[0].getText()
print(word_text)

type_html = soup.select('span.fl')
type_text = type_html[0].getText()
print(type_text)

# means all p inside of every div of class midbt
definition_html = soup.select('div.midbt > p')
definition_text = definition_html[0].getText()
print(definition_text)


# Openrice Webscraping
# https://www.openrice.com/en/hongkong/restaurants?where=Hung%20Hom&page=2
#
# HTML identifiers:
# name: h2.title-name > a
# Price: div.icon-info-food-price > span
# positives: div.smile-face > span
# negatives: div.sad-face > span

# Lists
names =[]
price = []
positives = []
negatives = []

folder_path = './photo/'
if os.path.exists(folder_path) == False:  # 判断文件夹是否已经存在
    os.makedirs(folder_path)  # 创建文件夹

location = input("Choose a location:")
index = 0
for pagenum in range(5):
    # choose webpage
    target = 'https://www.openrice.com/en/hongkong/restaurants?where=' + location + '&page=' + str(pagenum + 1)
    # request the page
    res = requests.get(target, headers = {'User-Agent': "Chrome/74.0.3729.169"})
    # if res.status_code == 200:
    #   # normal
    # elif res.status_code == 302:
    #   # break
    # else:
    #   print("Error.")

    soup = BeautifulSoup(res.text, features="html.parser")
    names_html = soup.select('h2.title-name > a')
    prices_html = soup.select('div.icon-info-food-price > span')
    positives_html = soup.select('div.smile-face > span')
    negatives_html = soup.select('div.sad-face > span')
    img_html = soup.select('div.pois-restaurant-list-cell-content-left-restaurant-photo')

    print(img_html)
    print(negatives_html)

    num_of_restaurants = len(names_html)

    for i in range(num_of_restaurants):
        img_data = img_html[i].attrs['style']
        img_data = img_data.split("'")
        html = requests.get(img_data[1], headers={"User-Agent":"Chrome/74.0.3729.169"})
        img_name = folder_path + str(index) + str("-") +str(i) + '.png'
        # 以byte形式将图片数据写入
        with open(img_name, 'wb') as file:
            file.write(html.content)
            file.flush()
        file.close()
        print('the %dth image downloaded complete' % (index))
        names.append(names_html[i].getText())
        price.append(prices_html[i].getText())
        positives.append(positives_html[i].getText())
        negatives.append(negatives_html[i].getText())
        index = index+1

print("Finished")

# # Write to csv file
# datafile = open(location + 'data.csv', 'w')
# datafile.write("Name, Price_range, Positive, Negative\n")
# for i in range(len(names)):
#     row = ','.join([names[i], price[i], positives[i], negatives[i]])
#     datafile.write(row)
#     datafile.write('\n')
#
# datafile.close()
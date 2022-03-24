# *- coding: utf-8 -*
import json
import re
import time
import requests
import numpy as np

from lxml import etree


def get_product_code():
    url_1 = 'http://cheman.chemnet.com/'
    url_1_nums = np.arange(1, 588)
    # http://cheman.chemnet.com/dict/cas/1.html
    with open('product_codes.txt', 'a') as f:
        for num in url_1_nums:
            try:
                url = url_1 + f'/dict/cas/{num}.html'
                ds = requests.post(url)
                element = etree.HTML(ds.text)
                url_list = element.xpath("//li[@class='w22']//a/@href")
                f.write(str(url_list))
                if (num+1) % 20 == 0:
                    time.sleep(15)
                else:
                    time.sleep(np.random.randint(2, 5))
            except:
                print(num)
                break


def get_product_value():

    cookies = {
        'nsunid': '3kkIMGIEgWPB42gAEFjzAg==',
    }

    headers = {
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    urls = eval(open('product_codes_ts.txt').read())
    with open('product_value.json', 'a') as f:
        for num, url in enumerate(urls):
            try:

                response = requests.get(f'http://cheman.chemnet.com/{url}',
                                        headers=headers,
                                        cookies=cookies,
                                        verify=False)
                element = etree.HTML(str(response.text))
                tr_list = element.xpath("//table[@class='text12']//tr")
                data = {}
                for tr in tr_list:
                    td_list = tr.xpath("./td//text()")
                    if td_list:
                        # 过滤数据
                        td_list = [i.strip().replace("：", "") for i in td_list if len(re.sub(r":;|\s", "", i)) > 0]
                        if len(td_list) > 1:
                            data[td_list[0]] = "".join(td_list[1:])
                        else:
                            # 获取图片信息
                            src_list = tr.xpath("./td//img/@src")
                            data[td_list[0]] = src_list[0] if src_list else None

                f.write(json.dumps(data, ensure_ascii=False))
                if num % 1000 == 0:
                    time.sleep(5)
            except:
                print(num)
                continue


def get_product_value_supply():

    cookies = {
        'nsunid': '3kkIMGIEgWPB42gAEFjzAg==',
    }

    headers = {
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    urls = eval(open('../dataset_kbqa_ci/spider_product/product_codes_ts.txt').read())
    with open('product_value_supply.json', 'a') as f:
        for num, url in enumerate(urls):
            try:
                response = requests.get(f'http://cheman.chemnet.com/{url}',
                                        headers=headers,
                                        cookies=cookies,
                                        verify=False)
                element = etree.HTML(response.text)
                item = {}
                name = element.xpath("//td[text()='中文名称：']/following-sibling::td//text()")
                item["中文名称"] = name[0] if name else None
                tr_list = element.xpath("//td[text()='物化性质：']/../../tr")
                temp_list = []
                for tr in tr_list:
                    data = [i for i in [re.sub(r"\s", "", i) for i in tr.xpath(".//text()")] if len(i) > 0]
                    data = ", ".join(data)
                    temp_list.append(data)

                key = None
                while len(temp_list) > 0:
                    value = temp_list.pop(0)
                    if "：" in value and len(value) < 6:
                        key = value.replace("：", "")
                        item[key] = []
                    else:
                        item[key].append(value)
                f.write(json.dumps(item, ensure_ascii=False))
                if num % 1000 == 0:
                    time.sleep(5)
            except:
                print(num)
                continue


def get_product_png():
    product_ds = eval(open('../dataset_kbqa_ci/spider_product/product_value_ts.json').read())
    for pds in product_ds:
        if '分子结构' in pds and '中文名称' in pds:
            png_url = pds['分子结构']
            png_ds = requests.get(png_url)
            try:
                png_path = open(f'../dataset_kbqa_ci/spider_product/product_png/{pds["中文名称"]}.png', 'wb')
                png_path.write(png_ds.content)
            except:
                continue


# get_product_png()
get_product_value_supply()
# url = 'http://images-a.chemnet.com/suppliers/chembase/124/1248.gif'
# img = requests.get(url)
# img_pth = 'test.jpg'
# ts = open(img_pth, 'wb')
# ts.write(img.content)

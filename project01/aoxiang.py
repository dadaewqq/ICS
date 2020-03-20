# -*- coding: utf-8 -*-

import requests
import getpass

# 教务系统登陆网址
nwpu = 'http://us.nwpu.edu.cn/eams/login.action'

# 用户信息保存名
user_info = '用户信息.txt'


# 获取账号密码
def gain(is_input=True):
    # 初次登陆则输入账号密码 否则读取信息
    if is_input:
        user_name = input("\n请输入用户名:")
        password = getpass.getpass('请输入密码:')
    else:
        try:
            user_name, password = open(user_info).read().split('\n')
        except FileNotFoundError:
            return save()
    # 返回输入或读取的账号密码
    return user_name, password


# 保存账号密码
def save():
    # 保存已输入信息，写入文件
    userName, password = gain(is_input=True)
    with open(user_info, 'w') as f:
        f.write(userName + '\n' + password)
        return userName, password


# 登陆
def login(username, password, headers):
    # 初次登陆或者读取已保存信息登陆
    if username is None or password is None:
        username, password = gain(is_input=False)

    dataLogin = {
        'username': username,
        'password': password,
        'session_locale': 'zh_CN',
    }
    # 实例化session 发送post请求模拟登陆
    magic = requests.session()
    magic.post(url=nwpu, data=dataLogin, headers=headers)
    return magic


# 发送get请求
def get(url, headers={}, cookies={}, username=None, password=None):
    # 检查账号密码 headers使用默认值
    magic = check(username, password, headers)
    # 以文本形式返回get请求的数据
    return magic.get(url, headers=headers, cookies=cookies).text


# 发送post请求
def post(url, headers={}, cookies={}, data={}, username=None, password=None):
    # 检查账号密码 headers使用默认值
    magic = check(username, password, headers)
    # 以文本形式返回post请求的数据
    return magic.post(url, headers=headers, cookies=cookies, data=data).text


# 检查账密
def check(username, password, header={}):
    # 尝试登陆
    magic = login(username, password, header)
    # 验证账号密码正确性
    url = magic.get('http://us.nwpu.edu.cn/eams/home!index.action').url

    if url.find('login') >= 0:
        raise ValueError('用户名或密码错误')
    return magic

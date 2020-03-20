# -*- coding: utf-8 -*-

import json
import re
import datetime
from lxml import etree
import aoxiang
import table


# 获取时间表
def get_time_table():
    time_table_js = [
        {
            "name": "1",
            "during": "45",
            "time": {
                "start": "0830"
            }
        },
        {
            "name": "2",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "3",
            "during": "45",
            "offset": "20"
        },
        {
            "name": "4",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "5",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "6",
            "during": "45",
            "offset": "0"
        },
        {
            "name": "7",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "8",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "9",
            "during": "45",
            "offset": "20"
        },
        {
            "name": "10",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "11",
            "during": "45",
            "time": {
                "start": "1900"
            }
        },
        {
            "name": "12",
            "during": "45",
            "offset": "10"
        },
        {
            "name": "13",
            "during": "45",
            "offset": "0"
        }
    ]
    time_table = [{}]
    for i in time_table_js:
        try:
            time_start = i.get('time').get('start')
        except AttributeError:
            time_start = None
        if time_start is None:
            time_start = time_table[-1].get('end')
            time_start = datetime.datetime(1, 1, 1, int(time_start[0:2]), int(time_start[2:4]), 0)
            time_start += datetime.timedelta(minutes=int(i.get('offset')))
        else:
            time_start = datetime.datetime(1, 1, 1, int(time_start[0:2]), int(time_start[2:4]), 0)
        time_end = time_start + datetime.timedelta(minutes=int(i.get('during')))

        time_table.append({
            "name": i.get('name'),
            "start": format(str(time_start.hour), '0>2') + format(time_start.minute, '0>2'),
            "end": format(str(time_end.hour), '0>2') + format(time_end.minute, '0>2'),
        })
    sch = {}
    for i in time_table[1:]:
        sch[i.get('name')] = {'start': i.get('start'), 'end': i.get('end')}
    return sch


# 登陆、获取课表
def attch(username=None, password=None):
    # 获取账号ids
    ids = aoxiang.get('http://us.nwpu.edu.cn/eams/courseTableForStd.action', username=username, password=password)
    ids = re.search('"ids","[0-9]+"', ids).group(0).split('"')[3]
    # 定义post内容
    info = aoxiang.post('http://us.nwpu.edu.cn/eams/courseTableForStd!courseTable.action', data={
        'ignoreHead': 1,
        'startWeek': 1,
        'semester.id': 37,
        'setting.kind': 'std',
        'project.id': 1,
        'ids': ids
    }, username=username, password=password)
    dic = {
        "星期一": "1",
        "星期二": "2",
        "星期三": "3",
        "星期四": "4",
        "星期五": "5",
        "星期六": "6",
        "星期日": "7",
    }
    # 获取时间表
    time_table = get_time_table()
    # 利用xpath进行元素过滤
    xpath = '/html/body/div/table/tbody'
    dom = etree.HTML(info, etree.HTMLParser())
    trs = len(dom.xpath(xpath + '/tr'))
    #       课程名称 安排 起止周 教师
    infoIndex = [4, 8, 9, 5]

    # 数据清洗 存入数组
    sch = []
    for i in range(trs):
        data = lambda y: \
            dom.xpath(xpath + '/tr[{}]/td[{}]//text()'.format(i + 1, infoIndex[y]))[0].strip()

        if data(3) == '在线开放课程':
            continue
        for c in dom.xpath(xpath + '/tr[{}]/td[{}]//text()'.format(i + 1, infoIndex[1])):
            teacher = data(3)
            c = c[c.find('星期'):].strip()
            name = data(0)

            infoList = c.split(' ')
            for j in infoList[2].split(','):
                time = infoList[1].split('-')
                week = j.replace('[', '').replace(']', '').split('-')

                sch.append({
                    "name": name,
                    "week": {
                        "start": week[0],
                        "end": week[-1]
                    },
                    "day": dic[infoList[0]],
                    "time": {
                        "start": time_table[time[0]].get('start'),
                        "end": time_table[time[1]].get('end'),
                    },
                    "room": infoList[-1],
                    "teacher": teacher,
                })
    return sch


# 主函数入口
sch = attch(username=None, password=None)

# 将课表信息写入文件
with open('课表/课表.json', 'w', encoding='utf8') as f1:
    f1.write(json.dumps(sch, ensure_ascii=False, indent=4))
    print("\njson格式课表已生成")

# 转换课表格式为ics格式
try:
    with open('课表/课表.json', encoding='utf8') as f2:
        j_s = json.loads(f2.read())
        edule = table.get_calendar(js=j_s, term_start='20200224', method=None)

        with open('课表/课表.ics', 'w', encoding='utf8') as f3:
            f3.write(edule)
            print("\n已转换为ics格式课表")
except KeyboardInterrupt:
    print('\nInterrupted')
    exit(0)

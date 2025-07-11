# config.py

# 创建中英文特征名称映射
FEATURE_NAME_MAPPING = {
    '水温(℃)': 'Water Temperature (℃)',
    'pH(无量纲)': 'pH (Dimensionless)',
    '溶解氧(mg/L)': 'Dissolved Oxygen (mg/L)',
    '电导率(μS/cm)': 'Conductivity (μS/cm)',
    '浊度(NTU)': 'Turbidity (NTU)',
    '高锰酸盐指数(mg/L)': 'Permanganate Index (mg/L)',
    '氨氮(mg/L)': 'Ammonia Nitrogen (mg/L)',
    '总磷(mg/L)': 'Total Phosphorus (mg/L)',
    '总氮(mg/L)': 'Total Nitrogen (mg/L)',
    '叶绿素α(mg/L)': 'Chlorophyll-α (mg/L)',
    '藻密度(cells/L)': 'Algae Density (cells/L)'
}

# 创建站点名称映射
STATION_NAME_MAPPING = {
    '201医院': '201 Hospital',
    '三块石': 'Three Stone',
    '三川': 'Three Rivers',
    '三谷庄': 'Sanguzhang',
    '三邑大桥': 'Sanyi Bridge',
    '上河坝': 'Upper Dam',
    '上石盘': 'Upper Stone Plate',
    '两汇水': 'Two Waters Junction',
    '两河': 'Two Rivers',
    '两河口': 'Two Rivers Estuary',
    '丰谷': 'Fenggu',
    '二江寺': 'Erjiang Temple',
    '倮果': 'Luoguo',
    '八庙沟': 'Eight Temple Valley',
    '八角': 'Bajiao',
    '凉姜沟': 'Liangjiang Valley',
    '北川通口': 'Beichuan Mouth',
    '升钟水库铁炉寺': 'Shengzhong Reservoir Tielu Temple',
    '南渡': 'Nandu',
    '县城马踏石点': 'County Town Matashi Point',
    '双江桥': 'Shuangjiang Bridge',
    '团堡岭': 'Tuanbao Ridge',
    '大埂': 'Dageng',
    '大安': 'Da\'an',
    '大岗山': 'Dagangshan',
    '大磨子': 'Damozi',
    '大蹬沟': 'Dadenggou',
    '太平渡': 'Taiping Ferry',
    '姚渡': 'Yaodu',
    '姜公堰': 'Jianggong Dam',
    '宏缘': 'Hongyuan',
    '小水沟': 'Small Water Ditch',
    '岗托桥': 'Gangtuo Bridge',
    '岳店子下': 'Lower Yuedianzi',
    '岷江大桥': 'Minjiang Bridge',
    '平武水文站': 'Pingwu Hydrological Station',
    '幸福村(河东元坝)': 'Xingfu Village (Hedong Yuanba)',
    '幸福村（河东元坝）': 'Xingfu Village (Hedong Yuanba)',
    '幺滩': 'Yaotan',
    '廖家堰': 'Liaojiayan',
    '彭山岷江大桥': 'Pengshan Minjiang Bridge',
    '悦来渡口': 'Yuelai Ferry',
    '手傍岩': 'Shoubangyan',
    '手爬岩': 'Shoupayan',
    '拦马山': 'Lanmashan',
    '拱城铺渡口': 'Gongchengpu Ferry',
    '挂弓山': 'Guagongshan',
    '文成镇': 'Wencheng Town',
    '昔街大桥': 'Xijie Bridge',
    '月波': 'Yuepo',
    '木城镇': 'Mucheng Town',
    '李家湾': 'Lijiawan',
    '李码头': 'Li Dock',
    '柏枝': 'Baizhi',
    '梓江大桥': 'Zijiang Bridge',
    '水磨沟村': 'Shuimogou Village',
    '江陵': 'Jiangling',
    '沙溪': 'Shaxi',
    '沱江大桥': 'Tuojiang Bridge',
    '沱江大桥(沱江二桥)': 'Tuojiang Bridge (Tuojiang Second Bridge)',
    '泸天化大桥': 'Lutianhua Bridge',
    '洛亥': 'Luohai',
    '洛须镇温托村': 'Luoxu Town Wentuo Village',
    '清平镇大庙村（摇金）': 'Qingping Town Damiao Village (Yaojin)',
    '清泉乡（文成镇）': 'Qingquan Township (Wencheng Town)',
    '清风峡': 'Qingfeng Gorge',
    '渭门桥': 'Weimen Bridge',
    '湾凼': 'Wandang',
    '溪口镇平桥村': 'Xikou Town Pingqiao Village',
    '烈面': 'Liemian',
    '球溪河口': 'Qiuxi Estuary',
    '白兔乡': 'Baitu Township',
    '百顷': 'Baiqing',
    '石门子': 'Shimenzi',
    '碳研所': 'Carbon Research Institute',
    '礼板湾(王妃岛)': 'Liban Bay (Wangfei Island)',
    '福田坝': 'Futian Dam',
    '红光村': 'Hongguang Village',
    '纳溪大渡口': 'Naxi Ferry',
    '联盟桥': 'Lianmeng Bridge',
    '胡市大桥': 'Hushi Bridge',
    '脚仙村': 'Jiaoxian Village',
    '舵石盘': 'Duoshipan',
    '苴国村': 'Juguo Village',
    '蔡家渡口': 'Caijia Ferry',
    '西平镇': 'Xiping Town',
    '象山': 'Xiangshan',
    '越溪河两河口': 'Yuexi River Two Rivers Estuary',
    '跑马滩': 'Paomatan',
    '车家河': 'Chejia River',
    '邛海湖心': 'Qionghai Lake Center',
    '都江堰水文站': 'Dujiangyan Hydrological Station',
    '醒觉溪': 'Xingjue Stream',
    '金子': 'Jinzi',
    '金沙江岗托桥': 'Jinsha River Gangtuo Bridge',
    '金溪电站': 'Jinxi Power Station',
    '阿七大桥': 'Aqi Bridge',
    '雅砻江口': 'Yalong River Estuary',
    '马尔邦碉王山庄': 'Ma\'erbang Diaowang Villa',
    '马边河河口': 'Mabian River Estuary',
    '鲁班岛': 'Luban Island',
    '黄龙溪': 'Huanglong Stream',
    '龙洞': 'Longdong',
    '龟都府': 'Guidufu'
}

# 获取特征的英文名称
def get_english_feature_name(chinese_name):
    return FEATURE_NAME_MAPPING.get(chinese_name, chinese_name)

# 获取站点的英文名称
def get_english_station_name(chinese_name):
    return STATION_NAME_MAPPING.get(chinese_name, chinese_name)
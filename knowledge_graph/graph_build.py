import json
import os
from tkinter import font
import pandas as pd
import dgl
from dgl.data.utils import save_graphs
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
from random import randint
import shutil
import traceback
import torch

sys.path.append('.')
from preprocessing.price_preprocess import new_left_company

#当共同的股份持有人超过下面个数是才连接边，领导人数同理
THRESHOLD = 1
# 图结构可视化的时候用到的节点个数
is_subgraph = 0
#注:两个nx图的节点数目是60


def get_company_json(kg_company_path):
    """获取用于知识图谱的公司信息

    Args:
        company_path (str): 公司数据的路径

    Returns:
        list: 公司信息的列表
    """
    with open(kg_company_path, 'r', encoding='utf-8') as f:
        company_detail = json.load(f)
        return company_detail


def get_company2use(company_left, company_detail):
    """获取深交所我们考虑的公司信息，而不是全部需要

    Args:
        company_left (list): 公司代码
        company_detail (list): 公司信息字典所组成的列表

    Returns:
        list: 需要用到的公司字典
    """
    #保留需要使用的公司dict
    company2use = []
    for data_dict in company_detail:
        if isinstance(data_dict['stock_id'], str):
            if data_dict['stock_id'] in company_left:
                company2use.append(data_dict)
            else:
                continue
        else:
            if str(data_dict['stock_id']) in company_left:
                company2use.append(data_dict)
            else:
                continue
    #做缓存 company.json
    json_data = json.dumps(company2use,
                           ensure_ascii=False).encode(encoding='utf-8')
    if os.path.exists('knowledge_graph\company_use.json'):
        print('已经有缓存文件!')
    else:
        with open('knowledge_graph\company_use.json', 'wb') as f:
            f.write(json_data)
            print('已保存相关结果。')
    return company2use


def get_index1(lst=None, item=''):
    """获取指定元素所有的索引

    Args:
        lst (list, optional): 原始的元素集合. Defaults to None.
        item (str, optional): 需要查找的元素. Defaults to ''.

    Returns:
        list: 下标的索引
    """
    return [index for (index, value) in enumerate(lst) if value == item]


def edge_subgraph(node_list, industry_index):
    """获取边的链接关系

    Args:
        node_list (list): 节点列表
        industry_index (list): 索引列表

    Returns:
        list: 边的起始节点和终止节点组成的元组
    """
    edge_list = []
    for index in range(len(industry_index)):
        for j in range(index + 1, len(industry_index)):
            edge_list.append((node_list[industry_index[index]],
                              node_list[industry_index[j]]))
    return edge_list


def build_graph(graph_type, is_visual, graph_vis_lable, is_subgraph,
                company_path, kg_company_path):
    """根据不同的要求进行知识图谱的构建

    Args:
        graph_type (str): 要构建数目样的知识图，比如行业，还是地区等
        is_visual (bool): 是否要进行可视化
        graph_vis_lable (str): 节点进行可视化时采取的标签
        is_subgraph (int):如果取值为0，则使用全图进行可视化，否则选取排名前is_subgraph的公司进行可视化 
        company_path (str): 留下来的公司csv数据的目录
        kg_company_path (str): 知识图谱数据的存储数据的路径

    Returns:
        nx.Graph: networkx类型的图
    """
    #获取公司的代码
    company_left = new_left_company(company_path)
    company_left = list(map(lambda x: x.replace('.csv', ''), company_left))
    #公司数量
    company_num = len(company_left)
    # 获取公司具体知识图谱相关的信息
    company_detail = get_company_json(kg_company_path)
    company2use = get_company2use(company_left, company_detail)
    #创建一个映射
    company_idx = OrderedDict()
    for index, company in enumerate(company2use):
        company_idx[index] = company
    #获取名称等信息，用于节点命名
    if is_subgraph:
        stock_name = [
            company['stock_name'] for index, company in company_idx.items()
        ][:is_subgraph]
        stock_id = [
            str(company['stock_id']) for index, company in company_idx.items()
        ][:is_subgraph]
        company_name = [
            company['company_name'] for index, company in company_idx.items()
        ][:is_subgraph]
        #获取行业领域等信息，用于边的链接
        industry_name = [
            company['industry_name'] for index, company in company_idx.items()
        ][:is_subgraph]
        area = [
            company['area_name'] for index, company in company_idx.items()
        ][:is_subgraph]
        share_holder = [
            company['share_holder'] for index, company in company_idx.items()
        ][:is_subgraph]
        leader = [company['leader']
                  for index, company in company_idx.items()][:is_subgraph]
    else:
        stock_name = [
            company['stock_name'] for index, company in company_idx.items()
        ]
        stock_id = [
            str(company['stock_id']) for index, company in company_idx.items()
        ]
        company_name = [
            company['company_name'] for index, company in company_idx.items()
        ]
        #获取行业领域等信息，用于边的链接
        industry_name = [
            company['industry_name'] for index, company in company_idx.items()
        ]
        area = [company['area_name'] for index, company in company_idx.items()]
        share_holder = [
            company['share_holder'] for index, company in company_idx.items()
        ]
        leader = [company['leader'] for index, company in company_idx.items()]
    print('处理好建立节点和边关系的列表了。')
    #创建一个图
    Graph = nx.Graph()
    # 判断可视化的时候，节点的表示类型
    if graph_vis_lable == 'stock_name':
        Graph.add_nodes_from(stock_name)
    elif graph_vis_lable == 'stock_id':
        Graph.add_nodes_from(stock_id)
    elif graph_vis_lable == 'company_name':
        Graph.add_nodes_from(company_name)
    elif graph_vis_lable == 'None':
        Graph.add_nodes_from([i for i in range(company_num)])
    print('完成节点的构建')
    #开始确定边的关系
    node_list = list(Graph.nodes)
    #print(len(node_list))
    #print('前十个节点为：')
    #print(node_list[:10])
    if graph_type == 'industry':
        # 依据行业进行作图
        industry_name_ = list(set(industry_name))
        for elem in industry_name_:
            industry_index = get_index1(industry_name, elem)
            edge_list = edge_subgraph(node_list, industry_index)
            Graph.add_edges_from(edge_list)
        #绘图
        if is_visual:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            plt.rcParams['font.sans-serif'] = ['SimHei']
            #这里绘制一个子图，不然没法显示全
            weight = [elem[1] for elem in list(Graph.degree)]
            cmap = plt.cm.get_cmap('Greens')
            nx.draw_random(
                Graph,
                #pos=nx.draw_random(Graph),这个参数会导致后面出现很多无关的点，下同
                node_color=weight,
                cmap=cmap,
                edge_color='cyan',
                font_color='k',
                with_labels=True)
            #plt.show()
            plt.savefig(
                os.path.join(
                    'knowledge_graph\company_connection_graph',
                    graph_type + ' based company connnection graph.pdf'))
    elif graph_type == 'area':
        # 依据地区进行作图
        area_ = list(set(area))
        for elem in area_:
            area_index = get_index1(area, elem)
            edge_list = edge_subgraph(node_list, area_index)
            Graph.add_edges_from(edge_list)
        #绘图
        if is_visual:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            plt.rcParams['font.sans-serif'] = ['SimHei']
            #这里绘制一个子图，不然没法显示全
            weight = [elem[1] for elem in list(Graph.degree)]
            cmap = plt.cm.get_cmap('Greens')
            nx.draw_random(
                Graph,
                #pos=nx.draw_random(Graph),
                node_color=weight,
                cmap=cmap,
                edge_color='cyan',
                font_color='k',
                with_labels=True)
            #plt.show()
            plt.savefig(
                os.path.join(
                    'knowledge_graph\company_connection_graph',
                    graph_type + ' based company connnection graph.pdf'))
    elif graph_type == 'share_holder':
        # 依据股份持有者进行作图
        for index1, share1 in enumerate(share_holder):
            for index2, share2 in enumerate(share_holder):
                if isinstance(share1, str):
                    share1 = share1.replace(
                        "'", '').lstrip('[').rstrip(']').split(
                            ',')  #这里要注意，dict里面的list不是list，而是字符串
                if isinstance(share2, str):
                    share2 = share2.replace(
                        "'", '').lstrip('[').rstrip(']').split(',')
                intersection_set = set(share1) & set(share2)  #取交集
                if index1 != index2 and len(
                        intersection_set
                ) >= THRESHOLD:  #当共有的股份所有者占有了相当一部分股份时，可以认为此时关联比较紧密，以下同
                    Graph.add_edge(
                        node_list[index1],
                        node_list[index2],
                        #weight=len(intersection_set)  由于考虑了行业，这个权重没法统一，所以就不用了，下同
                    )
        ####注，下面这一部分代码，是因为注意到有太多孤立点，所以考虑结合行业多连接一些边，下同
        edge_list = list(Graph.edges)
        industry_name_ = list(set(industry_name))
        for elem in industry_name_:
            industry_index = get_index1(industry_name, elem)
            edge_list += edge_subgraph(node_list, industry_index)
        edge_list = list(set(edge_list))
        Graph.add_edges_from(edge_list)
        ###
        #绘图
        if is_visual:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            plt.rcParams['font.sans-serif'] = ['SimHei']
            #这里绘制一个子图，不然没法显示全
            weight = [elem[1] for elem in list(Graph.degree)]
            cmap = plt.cm.get_cmap('Greens')
            nx.draw_random(
                Graph,
                #pos=nx.draw_random(Graph),
                node_color=weight,
                cmap=cmap,
                edge_color='cyan',
                font_color='k',
                with_labels=True)
            #plt.show()
            plt.savefig(
                os.path.join(
                    'knowledge_graph\company_connection_graph',
                    graph_type + ' based company connnection graph.pdf'))
    elif graph_type == 'leader':
        # 依据领导者进行作图
        for index1, share1 in enumerate(leader):
            for index2, share2 in enumerate(leader):
                if isinstance(share1, str):
                    share1 = share1.replace(
                        "'", '').lstrip('[').rstrip(']').split(
                            ',')  #这里要注意，dict里面的list不是list，而是字符串
                if isinstance(share2, str):
                    share2 = share2.replace(
                        "'", '').lstrip('[').rstrip(']').split(',')
                intersection_set = set(share1) & set(share2)
                if index1 != index2 and len(intersection_set) >= THRESHOLD:
                    Graph.add_edge(
                        node_list[index1],
                        node_list[index2],
                        #weight=len(intersection_set)
                    )
        ###
        edge_list = list(Graph.edges)
        industry_name_ = list(set(industry_name))
        for elem in industry_name_:
            industry_index = get_index1(industry_name, elem)
            edge_list += edge_subgraph(node_list, industry_index)
        edge_list = list(set(edge_list))
        Graph.add_edges_from(edge_list)
        ###
        #绘图
        if is_visual:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            plt.rcParams['font.sans-serif'] = ['SimHei']
            weight = [elem[1] for elem in list(Graph.degree)]
            cmap = plt.cm.get_cmap('Greens')
            #这里绘制一个子图，不然没法显示全
            nx.draw_random(
                Graph,
                #pos=nx.draw_random(Graph),
                node_color=weight,
                cmap=cmap,
                edge_color='cyan',
                font_color='k',
                with_labels=True)
            #plt.show()
            plt.savefig(
                os.path.join(
                    'knowledge_graph\company_connection_graph',
                    graph_type + ' based company connnection graph.pdf'))
    elif graph_type == None:
        #随机产生边
        src = randint(len(node_list),
                      size=randint(low=len(node_list),
                                   high=3 * len(node_list)))  #源节点
        dst = randint(len(node_list),
                      size=randint(low=len(node_list),
                                   high=3 * len(node_list)))  #目标节点
        Graph.add_edges_from(zip(src, dst))
        #绘图
        if is_visual:
            plt.rcParams.update({'figure.figsize': (12, 9)})
            plt.rcParams['font.sans-serif'] = ['SimHei']
            weight = [elem[1] for elem in list(Graph.degree)]
            cmap = plt.cm.get_cmap('Greens')
            #这里绘制一个子图，不然没法显示全
            nx.draw_random(
                Graph,
                #pos=nx.draw_random(Graph),
                node_color=weight,
                cmap=cmap,
                edge_color='cyan',
                font_color='k',
                with_labels=True)
            #plt.show()
            plt.savefig(
                os.path.join('knowledge_graph\company_connection_graph',
                             'random company connnection graph.pdf'))
    #返回nx的图，
    return Graph


def degree_plot(G, graph_name):
    """对图的度分布情况进行绘制

    Args:
        G (networkx.graph): networkx的图
    """
    degree = nx.degree_histogram(G)  #返回图中所有节点的度分布序列
    print(degree)
    x = range(len(degree))  #生成x轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]
    plt.figure(figsize=(5.8, 5.2), dpi=150)
    #将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.xlabel("节点的度", size=14)  # Degree
    plt.ylabel("频率", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.loglog(x, y, '.')  #在双对数坐标轴上绘制度分布曲线
    #plt.show()
    plt.savefig(
        os.path.join('knowledge_graph\company_connection_graph',
                     'Graph node histogram ' + graph_name + '.pdf'))
    print('保存成功！')


def statistics_graph(Graph, graph_name):
    """保存构建的图的统计信息

    Args:
        Graph (nx.networkx):  nx类型的图网络 
        graph_name (str): 保存文件的名字
    """
    graph_detail = OrderedDict()
    #部分指标取消了，是因为图并不联通
    print('边的数目为：')
    number_of_edges = Graph.number_of_edges()
    print(number_of_edges)
    graph_detail['number_of_edges'] = number_of_edges

    print('节点的个数为：')
    number_of_nodes = Graph.number_of_nodes()
    print(number_of_nodes)
    graph_detail['number_of_nodes'] = number_of_nodes

    print('群聚系数为：')
    avrage_clustering = nx.average_clustering(Graph)
    print(avrage_clustering)
    graph_detail['avrage_clustering'] = avrage_clustering

    # print('图的直径：')
    # diameter = nx.diameter(Graph)
    # print(diameter)
    # graph_detail['diameter'] = diameter

    # print('图的所有节点间平均最短路径长度：')
    # average_shortest_path_length = nx.average_shortest_path_length(Graph)
    # print(average_shortest_path_length)
    # graph_detail['average_shortest_path_length'] = average_shortest_path_length

    print('图的度中心性：')
    degree_centrality = nx.degree_centrality(Graph)
    print(degree_centrality)
    graph_detail['degree_centrality'] = degree_centrality

    node_degree = Graph.degree()
    print('最大度：\n')
    print(max(node_degree))
    print('最小度：\n')
    print(min(node_degree))
    graph_detail['min_degree'] = min(node_degree)
    graph_detail['max_degree'] = max(node_degree)
    #print('图的入度中心性：\n')
    #print(nx.in_degree_centrality(Graph))
    #print('图的出度中心性：\n')
    #print(nx.out_degree_centrality(Graph))

    # ==> Eccentricity
    # print('eccentricity:')
    # eccentricity = nx.eccentricity(Graph)
    # print(eccentricity)
    # graph_detail['eccentricity'] = eccentricity

    # ==> Radius
    # print('radius:')
    # radius = nx.radius(Graph)
    # print(radius)
    # graph_detail['radius'] = radius

    # ==> Periphery
    # print('periphery:')
    # periphery = nx.periphery(Graph)
    # print(periphery)
    # graph_detail['periphery'] = periphery

    # ==> Center
    # print('center:')
    # center = nx.center(Graph)
    # print(center)
    # graph_detail['center'] = center

    #b保存
    if os.path.exists(
            os.path.join('knowledge_graph\company_connection_graph',
                         'graph_detail_sta_' + graph_name + '.json')):
        print('已有缓存。')
    else:
        json_content = json.dumps(graph_detail,
                                  ensure_ascii=False).encode(encoding='utf-8')
        with open(
                os.path.join('knowledge_graph\company_connection_graph',
                             'graph_detail_sta_' + graph_name + '.json'),
                'wb') as f:
            f.write(json_content)
            print('读入完成！')


def move_file(src_path, dst_path, file):
    """文件移动

    Args:
        src_path (str): 源路径名
        dst_path (str): 目标路径名
        file (str): 文件名
    """
    try:
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        if os.path.exists(f_src):
            shutil.move(f_src, f_dst)
    except Exception as e:
        traceback.print_exc()


##todo 有点问题保存不了
def get_dgl_graph(is_subgraph):
    """图转化

    Args:
        is_subgraph(int): 是否使用全部节点进行作图
    Returns:
        DGL.graph: dgl的图
    """
    Graph1 = build_graph(
        'share_holder', False, 'stock_name', is_subgraph,
        'D:\毕设code\\news_data_from2016to2021\companies\\final_use',
        'D:\毕设code\knowledge_graph\company_use.json')
    Graph2 = build_graph(
        'leader', False, 'stock_name', is_subgraph,
        'D:\毕设code\\news_data_from2016to2021\companies\\final_use',
        'D:\毕设code\knowledge_graph\company_use.json')
    Graph1 = dgl.from_networkx(Graph1)
    Graph2 = dgl.from_networkx(Graph2)
    graph_labels = {'glabel': torch.tensor([0, 1])}
    save_graphs('D:\\毕设code\\data\\graph_.bin', [Graph1, Graph2], graph_labels)
    print('完成保存...')


def main():
    #处理一下名称当中带有ST和*ST的公司，这些公司曾经面临退市的风险
    shen_a = pd.read_excel(
        'news_data_from2016to2021\companies\深交所A股列表_主板.xlsx')
    for index in range(len(shen_a)):
        if 'ST' in shen_a.loc[index, 'A股简称'] or '*ST' in shen_a.loc[index,
                                                                    'A股简称']:
            move_file(
                'D:\毕设code\\news_data_from2016to2021\companies\\final_use',
                'D:\毕设code\\news_data_from2016to2021\companies\\ts_stock_price',
                str(shen_a.loc[index, 'A股代码']).zfill(6) + '.csv')
    print("已移除面临过退市风险的公司。")
    #正式建图
    Graph = build_graph(
        'share_holder', False, 'stock_name', is_subgraph,
        'D:\毕设code\\news_data_from2016to2021\companies\\final_use',
        'D:\毕设code\knowledge_graph\company.json')
    degree_plot(Graph, 'share_holder')
    statistics_graph(Graph, 'share_holder')


if __name__ == '__main__':
    #company = get_company_json('knowledge_graph\company.json')
    #print(len(company))
    #建议：只修改is_subgraph和graph_type即可
    # graph = build_graph(
    #     'share_holder', True, 'stock_name', is_subgraph,
    #     'D:\毕设code\\news_data_from2016to2021\companies\\final_use',
    #     'D:\毕设code\knowledge_graph\company.json')
    #main()
    graph = get_dgl_graph(0)
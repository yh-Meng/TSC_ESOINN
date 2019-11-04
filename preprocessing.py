"""-----------------------prep work---------------------"""
# import pandas as pd 
# import numpy as np 
# import matplotlib.pyplot as plt
# from esoinn import ESoinn


# def deal_array(arr):
# 	return arr

# # data_dir = './data/ECG200/ECG200_TRAIN.tsv'
# data_dir = './data/ECGFiveDays/ECGFiveDays_TRAIN.tsv'
# train_data = pd.read_csv(data_dir, sep='\t') # no head
# # print(test_data.index)
# train_data.index = pd.to_datetime(train_data.index)  # 将字符串索引转换成时间索引
# # print(test_data.index)
# # # 如果已有表头
# # test = pd.read_csv(data_dir, sep='\t', header=0)
# # # 如果有主列键
# # test = pd.read_csv('test.tsv', sep='\t', header=0, index_col='id')
# # print(type(train_data))
# # print(type(train_data.head)) # no head
# # print(type(train_data))

# # first_rows = train_data.head(10) # 返回前n条数据,默认返回5条
# # print(first_rows)
# # cols = train_data.columns # 返回全部列名
# # print(cols)
# # dimensison = train_data.shape # 返回数据的格式，数组，（行数，列数）
# # print('shape:', dimensison)
# # test_values = train_data.values # 返回底层的numpy数据
# # print(train_data.values.shape)
# # # print(train_data == test_data.values)
# # print(train_data.values[0].shape) # 一个样本


# # df = pd.Series(np.random.randn(600), index = pd.date_range('7/1/2016', freq = 'D', periods = 600))
# # r = df.rolling(window = 10)
# # print(r.mean())

# # dataframe.rolling 对每一个series滑动
# # r = pd.Series(train_data.values[0]).rolling(window=10) #  Data must be 1-dimensional
# # series_all = pd.DataFrame(train_data.values) 
# # print(series_all.shape) # ???
# # for win_szie in range(1, test_data.values.shape[0]+1):
# # 	series_n = pd.Series(train_data.values[:0])
# # 	# series_n = pd.DataFrame(train_data.values[0])
# # 	r = series_n.rolling(window=win_szie, min_periods=win_szie)
# # 	# r.max, r.median, r.std, r.skew, r.sum, r.var, r.allpy
# #   	# 通过rolling().apply()方法，可以在移动窗口上使用自己定义的函数。唯一需要满足的是，在数组的每一个片段上，函数必须产生单个值。
# #   	# DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
# # 	# print(r.mean()) 	
# # 	# print(r.var())
# # 	# if win_szie == 2:
# # 	# 	print(r.apply(deal_array, raw=True)) # not work,must reurn a ral number
# # 	# 	break

# # 	print(type(train_data.values[0]))

# # plt.figure(figsize=(15, 5))
# # # series_n.plot(style='r--')
# # # print(series_n.shape)
# # # # r.mean().plot(style='b')
# # # plt.show()

# def sliding_window(arr, w_size=10):
# 	'''
# 	param arr: 输入的列表/一维数组
# 	param w_size: 滑动窗口的长度
# 	return: 所有得到的子数组组成列表
# 	'''
# 	arrs = [0 for i in range(136)]
# 	l = len(arr)
# 	assert w_size <= l, 'window\'s size should be smaller than array\'s'
# 	for i in range(l):
# 		start_indx = i 
# 		end_indx = i + w_size 
# 		if end_indx <= l:
# 			arrs.append(arr[start_indx:end_indx]) # 把索引为start_indx到end_indx-1的元素加入
# 		else:
# 			return arrs


# pd.Series(train_data.values[5][1:]).plot(style='r-') # 数据集的data[i][0]是类别标记
# pd.Series(train_data.values[21][1:]).plot(style='b-')
# # pd.Series(train_data.values[0]).rolling(window=5).mean().plot(style='r--')
# # pd.Series(train_data.values[0]).rolling(window=5).max().plot(style='b--')	
# plt.show()

# # for i in range(22):
# # 	print(train_data.values[i][0])

# # train_data_num = train_data.values.shape[0] # 数据的样本的个数
# # train_data_len = train_data.values.shape[1] # 样本的长度
# # for i in range(train_data_num):
# # 	for j in range(1, train_data_len+1):
# # 		print(len(sliding_window(train_data.values[i], j)))




""" --------------------generate BCG dataset------------------------------- """ 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


def get_subsignal(data_dict, class_id, size=4000):
	keys = list(data_dict.keys())
	# print(keys[0])
	# os._exit(0)
	subsignals = []
	gap = 0
	begin = 4000
	end = begin + size
	for i in range(len(keys) - 3):
		while end < data_dict[keys[i+3]].shape[0]:
			subsignals.append(data_dict[keys[i+3]][begin:end, :])
			begin = end + gap
			end = begin + size
	subsignals = np.array(subsignals)
	class_ids = (np.ones((subsignals.shape[0], 1)) * class_id)
	return np.squeeze(np.insert(subsignals, 0, class_ids, axis=1))




data_jk = loadmat('./data/data_jk.mat')
data_nh = loadmat('./data/data_nh.mat')
data_nqa = loadmat('./data/data_nqa.mat') # type dict
data_yc = loadmat('./data/data_yc.mat') # type dict
print(data_jk.keys())

print(len(data_jk.keys())-3)
print(len(data_nh.keys())-3)
print(len(data_nqa.keys())-3)
print(len(data_yc.keys())-3)
print(len(data_jk.keys())-3 + len(data_nh.keys())-3 + len(data_nqa.keys())-3 + len(data_yc.keys())-3)
# dict_keys(['__header__', '__version__', '__globals__', 'g1', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15', 'g16', 'g17', 'g18', 'g19', 'g2', 'g20', 'g21', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'])
# print(data_nqa['g1'].shape)
# print(data_jk['__version__'])
# print(data_nh['__version__'])
# print(data_nqa['__version__'])
# print(data_yc['__version__'])

# t = [i for i in range(0, 4000, 1)]
# plt.plot(t, data_nqa['g1'][1000:1000+4000,:])
# plt.plot(t, data_nqa['g1'][1000+4000:1000+4000+4000,:])
# plt.show()
class_set = [0, 1, 2, 3]
cut_size = 4000
subsignals_jk = get_subsignal(data_dict=data_jk, class_id=class_set[0], size=cut_size)
subsignals_nh = get_subsignal(data_dict=data_nh, class_id=class_set[1], size=cut_size)
subsignals_nqa = get_subsignal(data_dict=data_nqa, class_id=class_set[2], size=cut_size)
subsignals_yc = get_subsignal(data_dict=data_yc, class_id=class_set[3], size=cut_size)
print(subsignals_jk.shape)
print(subsignals_nh.shape)
print(subsignals_nqa.shape)
print(subsignals_yc.shape)
samples = np.concatenate((subsignals_jk, subsignals_nh, subsignals_nqa, subsignals_yc), axis=0)
print(samples.shape) # (764, 4001)
# os._exit(0)
np.random.shuffle(samples)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(samples[:, 1:], samples[:, 0], test_size=0.9, random_state=1)
test_data = np.concatenate((y_test.reshape((-1,1)), X_test), axis=1)
train_data = np.concatenate((y_train.reshape((-1,1)), X_train), axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
train_data[:,1:] = scaler.fit_transform(train_data[:,1:])
test_data[:,1:] = scaler.transform(test_data[:,1:])
np.savetxt("./data/BCG4000/BCG_TRAIN.csv", train_data, delimiter=",")
np.savetxt('./data/BCG4000/BCG_TEST.csv', test_data, delimiter=",")
# os._exit(0)


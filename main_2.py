import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from esoinn import ESoinn
from soinn import Soinn
import random
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.feature_selection import SelectFromModel
import time


class SLBE(object):

	# 类的初始化
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.calsses = self.__read_data(data_dir)[0] # 样本类别
		self.data = self.__read_data(data_dir)[1]#[:,50:] # for Lightning7 # 样本数据，numpy数组
		self.data_num = self.data.shape[0] # 样本个数
		self.data_len = self.data.shape[1] # 样本的长度

	# 读取数据
	def __read_data(self, data_dir):
		'''h
		param data_dir: 数据所在路径
		return: 样本类别，样本数据
		'''
		data =  pd.read_csv(data_dir, sep='\t').values
		return list(set(data[:,0])), data


	# 归一化
	def normalization(self, data):
		_range = np.max(data) - np.min(data)
		return (data - np.min(data)) / _range

	# 标准化
	def standardization(self, data):
	    '''
	    param data: 二维数组，每一行为一个样本
	    '''
	    # print((data[:,0] != 0).all()) 
	    mu = np.mean(data, axis=1).reshape((-1, 1)) # 压缩列，对各行求均值，返回 m *1 矩阵
	    # print(data.shape)
	    # print(mu.shape)
	    # print(np.mean(data, axis=0).shape)

	    sigma = np.std(data, axis=1, ddof=1).reshape((-1, 1))
	    # ddof=1,无偏
	    # FaceFour数据集中有些子序列标准差为0

	    # print('sigma.shape', sigma.shape)
	    # for i in range(sigma.shape[0]):
	    # 	if 0. in sigma[i]:
	    # 		print(sigma[i])
	    # # print((data - mu).shape)
	    # os._exit(0)
	    # print(((data - mu) / sigma).shape)
	    # os._exit(0)
	    return (data - mu) / sigma


	# 滑动窗口，返回指定长度的带输入序列类别标记的所有长度为w_size+1的子序列
	def sliding_window(self, arr, w_size):
		'''
		param arr: 输入的列表/一维数组，第一个元素为类别信息，不做处理
		param w_size: 滑动窗口的长度
		return: 所有得到的子数组组成的列表
		'''
		sub_arrs = []
		l = self.data_len
		# print(l)
		# print(arr)
		assert w_size > 0 and w_size <= l, \
			'window\'s size should be bigger than 0 and smaller than array\'s'
		for i in range(1, l+1):
			sub_arr = [0 for i in range(w_size+1)]
			# sub_arr = [0 for i in range(l)]
			sub_arr[0] = arr[0] # 类别标记保留
			start_indx = i
			end_indx = i + w_size 
			if end_indx <= l:
				# 用arr中索引为start_indx到end_indx-1的元素替换掉arr_中的0得到子数组arr_
				# sub_arr[start_indx:end_indx] = arr[start_indx:end_indx] 
				sub_arr[1:end_indx-start_indx+1] = arr[start_indx:end_indx]
				sub_arrs.append(sub_arr) 
			else:
				# print(sub_arrs[-1])	
				# os._exit(0)
				return sub_arrs

	# 生成子序列，一批为训练样本的指定一个长度的所有子序列
	def subseries_generator(self, data, min_, max_):
		'''
		param data: 二维数组,第一列为类别信息，不做处理
		return: 二维数组，输入数组的子数组
		'''
		print('here')
		data_num = data.shape[0]
		data_len = data.shape[1]
		ws_min, ws_max  = int(data_len*min_), int(data_len*max_)
		ws = [i for i in range(ws_min, ws_max, ws_min)] # 窗口的各种长度
		print(ws)	
		while True:
			for ws_ in ws:
				print('ws_:', ws_)
				# subseries = np.zeros((1, ws_+1)) # 开个头
				subseries = np.zeros((1, ws_+1))
				for i in range(data_num):
					temp = self.sliding_window(data[i], ws_)
					# if np.nan in temp:
					# 	print()
					# print(len(temp)) # len(series)-ws_+1
					subseries =  np.append(subseries, np.array(temp), axis=0)
				subseries[1:,1:] = self.standardization(subseries[1:,1:]) # 子序列标准化
				# print(subseries[1:,1:][:5])
				# os._exit(0)
				print('yield')
				yield subseries[1:,1:]



	def candidate_by_random(self, cluster):
		'''
		param cluster: 经ESOINN得到的类簇，每个簇包含若干个shapelet
		return: 从每个簇中随机选取一个shapelet组成的列表
		'''
		pass


	def get_info_entropy(self):
		pass


	def candidate_by_infoGain(self, cluster):
		'''
		param cluster: 经ESOINN得到的类簇，每个簇包含若干个shapelet
		return: 根据信息增益从每个簇中随机选取一个shapelet组成的列表
		'''
		for i in self.calsses:
			pass

	# 特征转化
	def feature_conversion(self, series, all_shapelets, alpha):
		'''
		param series: 一个时间序列，一维数组 是否需归一化？？？
		param shapelets: 有待进行特征转换的shapelet集合，二维数组(一般有多个shapelet)
		param similarity_thresholds: 每个shapelet(训练完成后的ESOINN的节点)相应的相似度阈值
		param alpha: 人工指定的参数,alpha>0,alpha调节h的范围[min,max],min>0,max<=1
		'''
		sdist = []
		feature = []
		for shapelets in all_shapelets:
			sub_series = np.array(self.sliding_window(series, len(shapelets[0])-2))
			for shapelet in shapelets:
				# print(len(shapelet)-2)
				# '''for debugging'''
				# print('sub_series.shape', sub_series.shape) # for example:(124, 14)
				# print('sub_series.shape', sub_series[:,1:].shape) # for example:(124, 13)
				# # sdist是shapelet与T的跟shapelet长度相同的子序列的距离中的最小值
				# print(np.linalg.norm(sub_series[:,1:]-shapelet, axis=1).shape) # for example:(124,)
				# print(np.min(np.linalg.norm(sub_series[:,1:]-shapelet, axis=1)))
				# os._exit(0) 
				sdist.append(np.min(np.linalg.norm(sub_series[:,1:]-shapelet[:-2], axis=1)))
			# print(sdist)

			# sub_series = np.array(self.sliding_window(series, len(shapelets[0])-2))
			# print(len(shapelets[0])-2)
			# print(shapelets[:,:-2])
			# print(np.mean(shapelets[:,:-2], axis=0))
			# os._exit(0)
			# 先求sdist再除以阈值
			# 那相似度阈值也得求平均
			# dist = np.linalg.norm(sub_series[:,1:]-np.mean(shapelets[:, :-2], axis=0), axis=1)
			# min_idx = np.argmin(dist)
			# sdist = np.min(dist)


			# print(sdist)
			# print(shapelets[:,-2])
			# print(sdist/shapelets[:,-2])
			# os._exit(0)
			# 一个feature的维数是shapelets的个数
			feature.append(np.exp(-alpha * np.array(sdist)/shapelets[:,-2]))
			sdist = []
		# print(feature)
		# print(type(np.array(feature[0])))
		# print(len(np.array(feature[1])))
		# # feature = np.array(feature).flatten() # not work
		# print(feature)
		# print([num for elem in feature for num in elem])
		# os._exit(0)
		# print(feature)
		# return feature
			# print(shapelets[:,-2])
			# print(shapelets[:,-1])
		# os._exit(0)

		return [num for elem in feature for num in elem]


	# 以各类shapelet的平均作为候选shapelet
	def feature_conversion_(self, series, all_shapelets, alpha):
		'''
		param series: 一个时间序列，一维数组 是否需归一化？？？
		param shapelets: 有待进行特征转换的shapelet集合，二维数组(一般有多个shapelet)
		param similarity_thresholds: 每个shapelet(训练完成后的ESOINN的节点)相应的相似度阈值
		param alpha: 人工指定的参数,alpha>0,alpha调节h的范围[min,max],min>0,max<=1
		'''
		feature = []
		for shapelets in all_shapelets:
			sub_series = np.array(self.sliding_window(series, len(shapelets[0])-2))
			for class_id in list(set(shapelets[:, -1])):
				shapelets_cluster = []
				for shapelet in shapelets:
					# print(len(shapelets[0])-2)
					# print(shapelets[:,:-2])
					# print(np.mean(shapelets[:,:-2], axis=0))
					# os._exit(0)
					if shapelet[-1] == class_id:
						shapelets_cluster.append(shapelet)
					# 先求sdist再除以阈值
					# 那相似度阈值也得求平均
				shapelets_cluster = np.array(shapelets_cluster)
				sdist = np.min(np.linalg.norm(sub_series[:,1:]-np.mean(shapelets_cluster[:,:-2], axis=0), axis=1))
				feature.append(np.exp(-alpha * np.array(sdist)/np.mean(shapelets_cluster[:,-2])))
		# print(feature)
		# print(type(np.array(feature[0])))
		# print(len(np.array(feature[1])))
		# # feature = np.array(feature).flatten() # not work
		# print(feature)
		# os._exit(0)
		return feature



# Reference:https://blog.csdn.net/datoutong_/article/details/78813233
#           https://blog.csdn.net/Bryan__/article/details/51607215
#           https://blog.csdn.net/Katherine_Cai_7/article/details/81326548 about LinearSVC
class SVM(object):

	def __init__(self):
		'''
		Sklearn.svm.LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, 
			multi_class=’ovr’,fit_intercept=True, intercept_scaling=1, class_weight=None, 
			verbose=0, random_state=None, max_iter=1000)
		'''
		# self.svc = LinearSVC() # got accuracy of 100% in ECGFiveDays!!!
		# 如何实现l1惩罚和hinge损失配合？
		# self.svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False) # got accuracy of 100% in ECGFiveDays!!!
		self.svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False)
		# self.svc = LinearSVC(penalty='l2', loss='hinge', dual=True)
		# self.svc = SVC() # worse

	def loss_func(self):
		pass

	# no need to redefine
	def fit(self, X, Y):
		self.svc.fit(X, Y)

	def predict(self, X):
		self.svc.predict(X)

	def score(self, X, Y):
		pass



def classifier(min_list, max_list, alpha_list, decision_tree=False):
	# data_dir = './data/UCRArchive_2018/Adiac/Adiac_TRAIN.tsv' # 1
	# data_dir = './data/UCRArchive_2018/Beef/Beef_TRAIN.tsv' # 2
	# data_dir = './data/UCRArchive_2018/BeetleFly/BeetleFly_TRAIN.tsv' # 3
	# data_dir = './data/UCRArchive_2018/BirdChicken/BirdChicken_TRAIN.tsv' # 4
	# data_dir = './data/UCRArchive_2018/ChlorineConcentration/ChlorineConcentration_TRAIN.tsv' # 5
	# data_dir = './data/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv' # 6
	# data_dir = './data/UCRArchive_2018/DiatomSizeReduction/DiatomSizeReduction_TRAIN.tsv' # 7
	# data_dir = './data/UCRArchive_2018/DistalPhalanxOutlineAgeGroup/DistalPhalanxOutlineAgeGroup_TRAIN.tsv' # 8
	# data_dir = './data/UCRArchive_2018/DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TRAIN.tsv' # 9
	# data_dir = './data/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TRAIN.tsv' # 10
	# data_dir = './data/UCRArchive_2018/ECGFiveDays/ECGFiveDays_TRAIN.tsv' # 12
	# data_dir = './data/UCRArchive_2018/FaceFour/FaceFour_TRAIN.tsv' # 12
	# data_dir = './data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv' # 13
	data_dir = './data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TRAIN.tsv' # 14
	# data_dir = './data/UCRArchive_2018/Lightning7/Lightning7_TRAIN.tsv' # 15 # 前50个数据点值是一样的
	# data_dir = './data/UCRArchive_2018/MedicalImages/MedicalImages_TRAIN.tsv' # 16 
	# data_dir = './data/UCRArchive_2018/MoteStrain/MoteStrain_TRAIN.tsv' # 17 # 遇到很多标准差为0的子序列，还没完成
	# data_dir = './data/UCRArchive_2018/MP_Little/MP_Little_TRAIN.tsv' # 18 ???
	# data_dir = './data/UCRArchive_2018/SonyAIBORobotSurface1/SonyAIBORobotSurface1_TRAIN.tsv' # 24
	# data_dir = './data/UCRArchive_2018/Symbols/Symbols_TRAIN.tsv' # 25
	# data_dir = './data/UCRArchive_2018/SyntheticControl/SyntheticControl_TRAIN.tsv' # 26
	# data_dir = './data/UCRArchive_2018/Trace/Trace_TRAIN.tsv' # 27
	# data_dir = './data/UCRArchive_2018/TwoLeadECG/TwoLeadECG_TRAIN.tsv' # 28


	
	# test_data = pd.read_csv('./data/UCRArchive_2018/Adiac/Adiac_TEST.tsv', sep='\t').values # 1
	# test_data = pd.read_csv('./data/UCRArchive_2018/Beef/Beef_TEST.tsv', sep='\t').values # 2
	# test_data = pd.read_csv('./data/UCRArchive_2018/BeetleFly/BeetleFly_TEST.tsv', sep='\t').values # 3
	# test_data = pd.read_csv('./data/UCRArchive_2018/BirdChicken/BirdChicken_TEST.tsv', sep='\t').values # 4
	# test_data = pd.read_csv('./data/UCRArchive_2018/ChlorineConcentration/ChlorineConcentration_TEST.tsv', sep='\t').values # 5
	# test_data = pd.read_csv('./data/UCRArchive_2018/Coffee/Coffee_TEST.tsv', sep='\t').values # 6
	# test_data = pd.read_csv('./data/UCRArchive_2018/DiatomSizeReduction/DiatomSizeReduction_TEST.tsv', sep='\t').values # 7
	# test_data = pd.read_csv('./data/UCRArchive_2018/DistalPhalanxOutlineAgeGroup/DistalPhalanxOutlineAgeGroup_TEST.tsv', sep='\t').values # 8
	# test_data = pd.read_csv('./data/UCRArchive_2018/DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TEST.tsv', sep='\t').values # 9
	# test_data = pd.read_csv('./data/UCRArchive_2018/DistalPhalanxTW/DistalPhalanxTW_TEST.tsv', sep='\t').values[:,50:] # 10
	# test_data = pd.read_csv('./data/UCRArchive_2018/ECGFiveDays/ECGFiveDays_TEST.tsv', sep='\t').values # 11
	# test_data = pd.read_csv('./data/UCRArchive_2018/FaceFour/FaceFour_TEST.tsv', sep='\t').values # 12
	# test_data = pd.read_csv('./data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv', sep='\t').values # 13
	test_data = pd.read_csv('./data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TEST.tsv', sep='\t').values # 14
	# test_data = pd.read_csv('./data/UCRArchive_2018/Lightning7/Lightning7_TEST.tsv', sep='\t').values[:,50:] # 15 # 前50个数据点值是一样的
	# test_data = pd.read_csv('./data/UCRArchive_2018/MedicalImages/MedicalImages_TEST.tsv', sep='\t').values # 16
	# test_data = pd.read_csv('./data/UCRArchive_2018/MoteStrain/MoteStrain_TEST.tsv', sep='\t').values # 17 # 遇到很多标准差为0的子序列，还没完成
	# test_data = pd.read_csv('./data/UCRArchive_2018/MP_Little/MP_Little_TEST.tsv', sep='\t').values 18 
	# test_data = pd.read_csv('./data/UCRArchive_2018/SonyAIBORobotSurface1/SonyAIBORobotSurface1_TEST.tsv', sep='\t').values # 24
	# test_data = pd.read_csv('./data/UCRArchive_2018/Symbols/Symbols_TEST.tsv', sep='\t').values # 25
	# test_data = pd.read_csv('./data/UCRArchive_2018/SyntheticControl/SyntheticControl_TEST.tsv', sep='\t').values # 25
	# test_data = pd.read_csv('./data/UCRArchive_2018/Trace/Trace_TEST.tsv', sep='\t').values # 27
	# test_data = pd.read_csv('./data/UCRArchive_2018/TwoLeadECG/TwoLeadECG_TEST.tsv', sep='\t').values # 28


	# data_dir = './data/ECGFiveDays/ECGFiveDays_TRAIN.tsv' # 训练数据的所在路径
	# data_dir = './data/ECG200/ECG200_TRAIN.tsv'
	# data_dir = './data/ECG5000/ECG5000_TRAIN.tsv'
	# data_dir = './data/UCRArchive_2018/FaceFour/FaceFour_TRAIN.tsv'


	# test_data = pd.read_csv('./data/ECGFiveDays/ECGFiveDays_TEST.tsv', sep='\t').values
	# test_data = pd.read_csv('./data/ECG200/ECG200_TEST.tsv', sep='\t').values
	# test_data = pd.read_csv('./data/ECG5000/ECG5000_TEST.tsv', sep='\t').values
	# test_data = pd.read_csv('./data/UCRArchive_2018/FaceFour/FaceFour_TEST.tsv', sep='\t').values

	# print(set(test_data[:,0]))
	# os._exit(0)

	slbe = SLBE(data_dir)
	# for i in range(slbe.data.shape[0]):
	# 	if np.nan in slbe.data[i]:
	# 		print('nan')
	# os._exit(0)
	esoinn = ESoinn(delete_node_period=100) # iteration_threshold(去噪周期λ)设多少为好?
	
	if decision_tree:
		clf=tree.DecisionTreeClassifier()
	else:
		clf = SVM()

	file_name = './logs2/scores_ItalyPowerDemand_maxlist_DT.txt'
	f = open(file_name, 'w')
	f.write('min' + '\t')
	f.write('alpha' + '\t')
	f.write('accuracy' + '\t\t')
	f.write('time_cost(seconds)' + '\r\n')
	# os._exit(0)

	# parameter search
	best_score = 0
	for min_ in min_list:
		for alpha in alpha_list:
			for max_ in max_list:
				print('training esoinn...')
				ws_min, ws_max  = int(slbe.data_len*min_), int(slbe.data_len*max_) # alpha=0.1 for Adiac
				ws = [i for i in range(ws_min, ws_max, ws_min)] # 窗口的各种长度
				# print(ws)
				length_num = len(ws) # 子序列长度的种类数
				# print('length_num', length_num)
				gen = slbe.subseries_generator(slbe.data, min_, max_)
				triple = []
				# 对每种长度的子序列训练
				for i in range(length_num):
					esoinn.fit(next(gen), epochs=1) # 按长度的顺序分批独立对网络进行训练 epochs=50 may be too big
					# print(esoinn.nodes) # will be saved below
					print('number of nodes/shapelets: ', len(esoinn.nodes))
					
					nodes_st = esoinn.get_similarity_threshold(esoinn.nodes) # 节点的相似度阈值,should be checked
					
					# 拼接后ESOINN的节点权重、节点的相似度阈值和节点类别标签在一起
					triple.append(np.concatenate((esoinn.nodes, nodes_st.reshape((-1, 1)), \
						np.array(esoinn.node_labels).reshape((-1, 1))), axis=1))
				# np.save('./transitional_data/esoinn_nodes_st_labels_it100_max40.npy', np.array(triple))

				triple = np.array(triple)
				

				# 对训练样本进行特征转化
				features = []
				print('feature transforming...')
				# a long wait, terrible time complexity
				for i in range(slbe.data_num):
					features.append(slbe.feature_conversion(slbe.data[i], triple, alpha=alpha)) 
				# PCA降维会怎么样？
				features = np.array(features)
				print('feature\'s shape: ', features.shape)
				print('one feature: ', features[5])
				# print(cosine_similarity(features)) # 没有区分度
				# os._exit(0)

				print('classifier fitting...')
				y_train = slbe.data[:,0].astype(int)
				print(y_train)
				y_train[y_train==-1] = 2
				print(y_train)
				# np.random.shuffle(y_train)
				if decision_tree:
					clf.fit(features, y_train)
				else:
					clf.svc.fit(features, y_train)
					# model = SelectFromModel(clf.svc, prefit=True)
					# features_new = model.transform(features)
					# # print(features_new)
					# print('new feature shape:', features_new.shape)
					# os._exit(0)
					print(clf.svc.decision_function(features))
				# os._exit(0)

				features_test = []
				start = time.clock()
				print('predicting...')
				# predict samples
				
				# test_data[:,1:] = slbe.standardization(test_data[:,1:])
				# 对测试样本进行特征转换
				for i in range(test_data.shape[0]):
					features_test.append(slbe.feature_conversion(test_data[i], triple, alpha=alpha))
				# alpha is important!!!
				
				# print(cosine_similarity(features_test)) # 没有区分度
				# os._exit(0)
				# 测试，统计准确率
				y_test = test_data[:,0].astype(int)
				# y_test[y_test==-1] = 2
				# np.random.shuffle(y_test)
				if decision_tree:
					print(clf.predict(features_test))
					score = clf.score(features_test, y_test)
				else:
					print(clf.svc.predict(features_test))
					score = clf.svc.score(features_test, y_test)
				end = time.clock()
				time_cost = end-start
				print('predicting cost %f seconds' % time_cost)
				# model = SVC(gamma=gamma,C=C)
				print(score)
				if score > best_score:  #找到最好表现的参数
					best_score = score
					best_parameters = {'min':min_,'alpha':alpha}
				f.write(str(min_) + '\t')
				f.write(str(alpha) + '\t')
				f.write(str(score) + '\t')
				f.write(str(time_cost) + '\r\n')
	f.close()
	esoinn.f.close()
	print('on dataset of ', data_dir)
	print('best params:', best_parameters)
	print('best score:', best_score)



def test(min_, alpha, decision_tree=False):
	# data_dir = './data/UCRArchive_2018/Adiac/Adiac_TRAIN.tsv' # 1
	# data_dir = './data/UCRArchive_2018/Beef/Beef_TRAIN.tsv' # 2
	# data_dir = './data/UCRArchive_2018/BeetleFly/BeetleFly_TRAIN.tsv' # 3
	# data_dir = './data/UCRArchive_2018/BirdChicken/BirdChicken_TRAIN.tsv' # 4
	# data_dir = './data/UCRArchive_2018/ChlorineConcentration/ChlorineConcentration_TRAIN.tsv' # 5
	# data_dir = './data/UCRArchive_2018/Coffee/Coffee_TRAIN.tsv' # 6
	# data_dir = './data/UCRArchive_2018/DiatomSizeReduction/DiatomSizeReduction_TRAIN.tsv' # 7
	# data_dir = './data/UCRArchive_2018/DistalPhalanxOutlineAgeGroup/DistalPhalanxOutlineAgeGroup_TRAIN.tsv' # 8
	# data_dir = './data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv' # 9
	# data_dir = './data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TRAIN.tsv' # 10
	# data_dir = './data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv' # 13
	# data_dir = './data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TRAIN.tsv' # 14
	# data_dir = './data/UCRArchive_2018/Lightning7/Lightning7_TRAIN.tsv' # 15 # 前50个数据点值是一样的
	# data_dir = './data/UCRArchive_2018/MedicalImages/MedicalImages_TRAIN.tsv' # 16 
	# data_dir = './data/UCRArchive_2018/MoteStrain/MoteStrain_TRAIN.tsv' # 17 # 遇到很多标准差为0的子序列，还没完成
	# data_dir = './data/UCRArchive_2018/MP_Little/MP_Little_TRAIN.tsv' # 18 ???
	data_dir = './data/UCRArchive_2018/SonyAIBORobotSurface1/SonyAIBORobotSurface1_TRAIN.tsv' # 24
	# data_dir = './data/UCRArchive_2018/Symbols/Symbols_TRAIN.tsv' # 25
	# data_dir = './data/UCRArchive_2018/SyntheticControl/SyntheticControl_TRAIN.tsv' # 26
	# data_dir = './data/UCRArchive_2018/Trace/Trace_TRAIN.tsv' # 27
	# data_dir = './data/UCRArchive_2018/TwoLeadECG/TwoLeadECG_TRAIN.tsv' # 28


	
	# test_data = pd.read_csv('./data/UCRArchive_2018/Adiac/Adiac_TEST.tsv', sep='\t').values # 1
	# test_data = pd.read_csv('./data/UCRArchive_2018/Beef/Beef_TEST.tsv', sep='\t').values # 2
	# test_data = pd.read_csv('./data/UCRArchive_2018/BeetleFly/BeetleFly_TEST.tsv', sep='\t').values # 3
	# test_data = pd.read_csv('./data/UCRArchive_2018/BirdChicken/BirdChicken_TEST.tsv', sep='\t').values # 4
	# test_data = pd.read_csv('./data/UCRArchive_2018/ChlorineConcentration/ChlorineConcentration_TEST.tsv', sep='\t').values # 5
	# test_data = pd.read_csv('./data/UCRArchive_2018/Coffee/Coffee_TEST.tsv', sep='\t').values # 6
	# test_data = pd.read_csv('./data/UCRArchive_2018/DiatomSizeReduction/DiatomSizeReduction_TEST.tsv', sep='\t').values # 7
	# test_data = pd.read_csv('./data/UCRArchive_2018/DistalPhalanxOutlineAgeGroup/DistalPhalanxOutlineAgeGroup_TEST.tsv', sep='\t').values # 8
	# test_data = pd.read_csv('./data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TEST.tsv', sep='\t').values # 9
	# test_data = pd.read_csv('./data/UCRArchive_2018/Lightning7/Lightning7_TEST.tsv', sep='\t').values[:,50:] # 10
	# test_data = pd.read_csv('./data/UCRArchive_2018/GunPoint/GunPoint_TEST.tsv', sep='\t').values # 13
	# test_data = pd.read_csv('./data/UCRArchive_2018/ItalyPowerDemand/ItalyPowerDemand_TEST.tsv', sep='\t').values # 13
	# test_data = pd.read_csv('./data/UCRArchive_2018/Lightning7/Lightning7_TEST.tsv', sep='\t').values[:,50:] # 15 # 前50个数据点值是一样的
	# test_data = pd.read_csv('./data/UCRArchive_2018/MedicalImages/MedicalImages_TEST.tsv', sep='\t').values # 16
	# test_data = pd.read_csv('./data/UCRArchive_2018/MoteStrain/MoteStrain_TEST.tsv', sep='\t').values # 17 # 遇到很多标准差为0的子序列，还没完成
	# test_data = pd.read_csv('./data/UCRArchive_2018/MP_Little/MP_Little_TEST.tsv', sep='\t').values 18 
	test_data = pd.read_csv('./data/UCRArchive_2018/SonyAIBORobotSurface1/SonyAIBORobotSurface1_TEST.tsv', sep='\t').values # 24
	# test_data = pd.read_csv('./data/UCRArchive_2018/Symbols/Symbols_TEST.tsv', sep='\t').values # 25
	# test_data = pd.read_csv('./data/UCRArchive_2018/SyntheticControl/SyntheticControl_TEST.tsv', sep='\t').values # 25
	# test_data = pd.read_csv('./data/UCRArchive_2018/Trace/Trace_TEST.tsv', sep='\t').values # 27
	# test_data = pd.read_csv('./data/UCRArchive_2018/TwoLeadECG/TwoLeadECG_TEST.tsv', sep='\t').values # 28

	slbe = SLBE(data_dir)
	# for i in slbe.data.shape[0]:
	# 	if np.nan in slbe.data[i]:
	# 		print('nan')
	# os._exit(0)
	esoinn = ESoinn(delete_node_period=100) # iteration_threshold(去噪周期λ)设多少为好?
	# esoinn = Soinn(delete_node_period=100)
	# gen = slbe.subseries_generator(slbe.data)
	# next(gen)

	print('training esoinn...')
	ws_min, ws_max  = int(slbe.data_len*min_), int(slbe.data_len*0.4)
	ws = [i for i in range(ws_min, ws_max, ws_min)] # 窗口的各种长度
	print(ws)
	length_num = len(ws) # 子序列长度的种类数
	print('length_num', length_num)
	print('ws_min:', ws_min)
	gen = slbe.subseries_generator(slbe.data, min_=min_)

	triple = []
	# 对每种长度的子序列训练
	for i in range(length_num):
		print('into')
		esoinn.fit(next(gen), epochs=1) # 按长度的顺序分批独立对网络进行训练
		print(esoinn.nodes) # will be saved below
		print('out')
		print('number of nodes/shapelets: ', len(esoinn.nodes))
		
		nodes_st = esoinn.get_similarity_threshold(esoinn.nodes) # 节点的相似度阈值,should be checked
		
		# 拼接后ESOINN的节点权重、节点的相似度阈值和节点类别标签在一起
		triple.append(np.concatenate((esoinn.nodes, nodes_st.reshape((-1, 1)), \
			np.array(esoinn.node_labels).reshape((-1, 1))), axis=1))
		# np.save('./transitional_data/soinn_nodes_st_labels_FaceFour.npy', np.array(triple))
	esoinn.f.close()

	triple = np.array(triple)
	# triple = np.load('./transitional_data/soinn_nodes_st_labels_ECGFiveDays_.npy', allow_pickle=True)
	# triple = np.load('./transitional_data/esoinn_nodes_st_labels_bcg.npy', allow_pickle=True)
	# print(triple.shape)
	# print(cosine_similarity(triple[1][:, :-2])) # 区分度良好
	# os._exit(0)
	for x in range(triple.shape[0]):
		for i in triple[x][:,-2]:
			if i==0:
				print(i)
	# os._exit(0)
	# triple = np.load('./transitional_data/esoinn_nodes_st_labels.npy')
	# print(type(triple))
	# print(triple[0].shape)
	# print(triple[1].shape)
	# os._exit(0)

	# print(triple[:,:-2]) # 节点权重
	# print(triple[1][:,-2]) # 节点相似度阈值
	# print(slbe.data[:,0].astype(int))
	# os._exit(0)
	# print(triple[:,-1]) # 节点类别

	# 对训练样本进行特征转化
	features = []
	print('feature transforming...')
	# a long wait, terrible time complexity
	for i in range(slbe.data_num):
		features.append(slbe.feature_conversion(slbe.data[i], triple, alpha=alpha)) # alpha设多少为好？
	# PCA降维会怎么样？
	features = np.array(features)
	print('old feature\'s shape:', features.shape)
	# print('one feature\' shape: ', features.shape)
	# print('one feature: ', features[5])
	# print(cosine_similarity(features)) # 没有区分度
	# os._exit(0)

	if decision_tree:
		clf=tree.DecisionTreeClassifier()
	else:
		clf = SVM()
	print('classifier fitting...')
	y_train = slbe.data[:,0].astype(int)
	print(y_train)
	y_train[y_train==-1] = 2
	print(y_train)
	# np.random.shuffle(y_train)
	if decision_tree:
		clf.fit(features, y_train)
	else:
		clf.svc.fit(features, y_train)
		# model = SelectFromModel(clf.svc, prefit=True)
		# features_new = model.transform(features)
		# print(features_new)
		# os._exit(0)
		print(clf.svc.decision_function(features))	

	
		print('old feature\'s shape:', features_new.shape)
		print(clf.svc.coef_)
	# os._exit(0)
	# print(clf.svc.decision_function(features))
	# os._exit(0)

	features_test = []
	print('predicting...')
	# predict samples
	
	# test_data[:,1:] = slbe.standardization(test_data[:,1:])
	# 对测试样本进行特征转换
	for i in range(test_data.shape[0]):
		features_test.append(slbe.feature_conversion(test_data[i], triple, alpha=alpha))
	# alpha is important!!!
	
	# print(cosine_similarity(features_test))
	# os._exit(0)
	# 测试，统计准确率
	# features_test_new = model.transform(features_test)
	# print(features_test_new.shape)

	y_test = test_data[:,0].astype(int)
	y_test[y_test==-1] = 2
	# np.random.shuffle(y_test)
	if decision_tree:
		print(clf.predict(features_test))
		score = clf.score(features_test, y_test)
	else:
		print(clf.svc.predict(features_test))
		score = clf.svc.score(features_test, y_test)
	print('accuracy:', score)
	# print(clf.svc.decision_function(features_test)) # 各样本点到超平面之间的距离



if __name__ == '__main__':
	# min_list = [0.1, 0.12, 0.15, 0.18, 0.2]#, 0.15, 0.2]
	min_list = [0.2, 0.25, 0.3, 0.35, 0.38] # for ItalyPowerDemand
	## min_list = [0.08, 0.09, 0.1, 0.11]
	max_list = [0.4, 0.5, 0.6, 0.7, 0.8]
	# max_list = [0.4]
	# alpha_list = [0.7, 0.8, 0.9, 1.0, 1.1]
	# alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5] # for Adiac
	alpha_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.0, 1.1]
	# alpha_list = [0.95, 0.97, 1.0, 1.02, 1.06, 1.1]#,0.8, 1.0, 1.2] # for TwoLeadECG
	## alpha_list = [1.0]
	classifier(min_list, max_list, alpha_list, decision_tree=True)

	# test(min_=0.11, alpha=1.2, decision_tree=True)

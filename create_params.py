#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Prints files with different combinations of parameters to be used to define different models.
'''

step_size=[5, 10]
num_cycles=[3,4]
kernel_size = [3,9,15]
filters = [10,30,60]
dilation_rate = [3,6]
alpha = [5,10,15]
batch_size=[16,32,48]
max_lr = 0.01
find_lr = 1

for s in step_size:
	for c in num_cycles:
		for k in kernel_size:
			for fil in filters:
				for dr in dilation_rate:
					for a in alpha:
						for b in batch_size:
							name = str(s)+'_'+str(c)+'_'+str(k)+'_'+str(fil)+'_'+str(dr)+'_'+str(a)+'_'+str(b)+'.params'
							with open(name, "w") as file:
								file.write('step_size='+str(s)+'\n')
								file.write('num_cycles='+str(c)+'\n')
								file.write('kernel_size='+str(k)+'\n')
								file.write('filters='+str(fil)+'\n')
								file.write('dilation_rate='+str(dr)+'\n')
								file.write('alpha='+str(a)+'\n')
								file.write('batch_size='+str(b)+'\n')
								file.write('max_lr='+str(max_lr)+'\n')
								file.write('find_lr='+str(find_lr)+'\n')


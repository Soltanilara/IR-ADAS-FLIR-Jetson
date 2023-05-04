import matplotlib.pyplot as plt
import numpy as np

def color_box(size=50):
    cmap = plt.get_cmap('jet')
    color_list = [
	[0,1,0], 
	[1, 1, 0],
	[0, 0, 1]	
	]

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3]) 

    return np.array(color_list)

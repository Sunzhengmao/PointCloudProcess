## off_to_ply.py
可以把ModelNet数据集中的.off文件批量转换成.ply文件，以方便后面的功能加载；如果有些同学下载的不是.off格式，而是.npy或其他格式，则不需要执行此文件。
## pca_normal.py
实现PCA分析和法向量计算该文件需要输入数据集文件路径，执行时它会自动加载数据文件，并完成PCA和法向量的计算，计算之后显示点云。
## voxel_filter.py
实现体素滤波该文件需要输入数据集文件路径，执行时会自动加载数据文件，并进行滤波，并显示滤波后的点云。
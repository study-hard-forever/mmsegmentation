from mmseg.apis import MMSegInferencer

'''模型推理
# Load models into memory
inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
# Inference
# inferencer('demo/demo.png', show=True)  # 直接显示可视化结果

# Save visualized rendering color maps and predicted results
# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# to save visualized rendering color maps and predicted results
# out_dir是保存输出结果的目录，img_out_dir和pred_out_dir是out_dir的子目录，用于保存可视化的渲染颜色映射和预测结果
inferencer('demo/demo.png', out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')  # 将结果保存到文件夹下
'''

# models is a list of model names, and them will print automatically
models = MMSegInferencer.list_models('mmseg')
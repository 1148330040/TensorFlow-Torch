# *- coding: utf-8 -*
import onnx
import tf2onnx.convert
import onnxruntime as ort


# 进入linux系统-mnt/tang_nlp/Knowledge_atlas_Q&A/运行下列命令转换onnx模型
# 其中model4robert
# python -m tf2onnx.convert  --saved-model save_model_best/  --output test.onnx  --opset 11
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('../models/best.onnx')

# convert model
model_simp, check = simplify(model)

print(check)
onnx.save(model_simp, '../models/best_simp.onnx')

# use model_simp as a standard ONNX model object

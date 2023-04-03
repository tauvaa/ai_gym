import json
import tensorflow as tf
from ai_utils.test_models import run_sim
import os


with open(os.path.join(os.path.split(os.path.dirname(__file__))[0], "model_info.json")) as f:
    models = json.loads(f.read())

models = list(zip(models.values(), models.keys()))

models.sort(key=lambda x: x[0], reverse=True)
if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    for m in models[0:5]:
        model = tf.keras.saving.load_model(m[1])
        avg = run_sim(model, 5, render_mode="human")
        print(f"average was: {avg}")

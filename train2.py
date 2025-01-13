import train
import dill
make_inference_fn, params, _= train.train_fn(environment=train.env,
                                       progress_fn=train.progress,
                                       eval_env=train.eval_env)

train.model.save_params("walk_policy", params)

with open("inference_fn", 'wb') as f:
    dill.dump(make_inference_fn, f)
import jax.numpy as jnp
from jax import jit as jjit
from functools import partial
from jax import vmap
from jax.lax import cond, while_loop
from jax import devices 
devices("gpu")
############自作モデルの実装##########

#誤差を求める関数

@partial(jjit, device=devices("gpu")[0])
def rmse(y, p):
    return jnp.sqrt(((y - p)**2).mean())

# rmse = jjit(rmse,device=devices[0])

#予測する関数(１行)
@partial(jjit, device=devices("gpu")[0])
def predict(row, model_x, model_y, prm,T,replace_value):
    t = jnp.abs(model_x - row)
    w = (1/2)**(t * prm / T)
    w = jnp.prod(w, axis=1)
    p = jnp.average(model_y, weights=w)
    p = jnp.nan_to_num(p, replace_value)
    return p

#予測する関数（全行）
@partial(jjit, device=devices("gpu")[0])
def predict_array(x, model_x, model_y, prm, T,replace_value):
    return vmap(lambda row : predict(row, model_x, model_y, prm, T, replace_value))(x)

#学習関数modelとtunerを引数で設定する必要がある（ランダムな要素を排除）
@partial(jjit, device=devices("gpu")[0])
def fit(T, model_x, model_y, tuner_x, tuner_y, replace_value, rate = 0.7):
    #初期状態の予測値・誤差・半減期
    same_cnt = 0 #精度が改善できなかった連続回数
    i = 0
    row,col = model_x.shape
    prm = jnp.full(col, 0.0)
    tuner_p = predict_array(tuner_x, model_x, model_y, prm, T, replace_value)
    err = rmse(tuner_y, tuner_p)
    #半減期と学習率の更新
    def update_loop(params):
        i, err, prm, same_cnt = params
        err_ex = err
        #半減期を縮小
        # T_ = T.at[i%col].set(T[i%col] * rate)
        prm_ = cond(prm[i%col] == 0.0,lambda:  prm.at[i%col].set(1.0), lambda: prm.at[i%col].set(prm[i%col] / rate))
        tuner_p = predict_array(tuner_x, model_x, model_y, prm_, T, replace_value)
        err_ = rmse(tuner_y, tuner_p)
        err, prm = cond(err_ < err, lambda: [err_, prm_], lambda: [err, prm])

        same_cnt =  cond(err_ex == err, lambda: same_cnt+1, lambda: 0)
        i+=1
        params = [i, err, prm, same_cnt]
        return params
    params = [i, err, prm, same_cnt]
    params = while_loop(lambda params: params[3] < col, update_loop, params)
    [i, err, prm, same_cnt] = params
    return {"prm": prm, "err": err, "try_cnt": i+1}

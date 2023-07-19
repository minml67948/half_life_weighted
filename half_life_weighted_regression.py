import jax.numpy as jnp
from jax import jit as jjit
from functools import partial
from jax import vmap
from jax.lax import cond, while_loop
from jax import devices
############自作モデルの実装##########

#誤差を求める関数

@partial(jjit)
def rmse(y, p):
    return jnp.sqrt(((y - p)**2).mean())


#重みの計算
@partial(jjit)
def weight(row, model_x, prm,T):
    t = jnp.abs(model_x - row)
    w = (1/2)**(t * prm / T)
    w = jnp.prod(w, axis=1)
    return w

#予測する関数(１行)
@partial(jjit)
def predict(row, model_x, model_y, prm,T):
    w = weight(row, model_x, prm,T)
    p = jnp.average(model_y, weights=w)
    return p

#予測する関数（全行）
@partial(jjit)
def predict_array(x, model_x, model_y, prm, T,replace_value):
    p = vmap(lambda row : predict(row, model_x, model_y, prm, T))(x)
    p = jnp.nan_to_num(p, replace_value)
    return p

#学習関数modelとtunerを引数で設定する必要がある（ランダムな要素を排除）
@partial(jjit)
def fit(T, model_x, model_y, tuner_x, tuner_y, replace_value, rate = 0.7):
    #初期状態の予測値・誤差・半減期
    same_cnt = 0 #精度が改善できなかった連続回数
    i = 0
    n = 0
    row,col = model_x.shape
    prm = jnp.full(col, 0.0)
    tuner_p = predict_array(tuner_x, model_x, model_y, prm, T, replace_value)
    err = rmse(tuner_y, tuner_p)
    #半減期と学習率の更新
    def update_loop(params):
        i, n, err, prm, same_cnt = params
        #半減期を縮小
        prm_ = cond(prm[n] == 0.0,lambda:  prm.at[n].set(1.0), lambda: prm.at[n].set(prm[n] / rate))
        tuner_p = predict_array(tuner_x, model_x, model_y, prm_, T, replace_value)
        err_ = rmse(tuner_y, tuner_p)

        #精度が改善すれば更新
        err_next, prm_next, same_cnt_next, n_next = cond(err_ < err, lambda: [err_, prm_, 0, n], lambda: [err, prm, same_cnt+1, (n+1)%col])

        i_next = i+1
        params_next = [i_next, n_next, err_next, prm_next, same_cnt_next]
        return params_next
    params = [i, n, err, prm, same_cnt]
    params_result = while_loop(lambda params: params[-1] < col, update_loop, params)
    [i_result, n_result, err_result, prm_result, same_cnt_result] = params_result
    return {"prm": prm_result, "err": err_result, "try_cnt": i_result+1}


############自作モデルの実装index##########

def make_index(points, model_x, model_y, prm, T, class_size):
    index = vmap(lambda point : weight(point, model_x, prm,T).argsort()[class_size:-1])(points)
    return model_x[index], model_y[index]


@partial(jjit)
def near_class(row, model_x_index, model_y_index, points, prm, T):
    nearest_point = weight(row, points, prm, T).argmax()
    nearest_class_x = model_x_index[nearest_point]
    nearest_class_y = model_y_index[nearest_point]
    return nearest_class_x, nearest_class_y

#予測する関数(１行)
@partial(jjit)
def predict_index(row, model_x_index, model_y_index, points, prm, T):
    model_x, model_y = near_class(row, model_x_index, model_y_index, points, prm, T)
    w = weight(row, model_x, prm,T)
    p = jnp.average(model_y, weights=w)
    return p

#予測する関数（全行）

@partial(jjit)
def predict_array_index(x, model_x_index, model_y_index, points, prm, T, replace_value):
    p = vmap(lambda row : predict_index(row, model_x_index, model_y_index, points, prm, T))(x)
    p = jnp.nan_to_num(p, replace_value)
    return p

#学習関数
@partial(jjit)
def fit_index(T, model_x_index, model_y_index, points, tuner_x, tuner_y, replace_value, rate = 0.7):
    #初期状態の予測値・誤差・半減期
    same_cnt = 0 #精度が改善できなかった連続回数
    i = 0
    n = 0
    row,col = model_x_index[0].shape
    prm = jnp.full(col, 0.0)
    tuner_p = predict_array_index(tuner_x, model_x_index, model_y_index, points, prm, T, replace_value)
    err = rmse(tuner_y, tuner_p)
    #半減期と学習率の更新
    def update_loop(params):
        i, n, err, prm, same_cnt = params
        #半減期を縮小
        prm_ = cond(prm[n] == 0.0,lambda:  prm.at[n].set(1.0), lambda: prm.at[n].set(prm[n] / rate))
        tuner_p = predict_array_index(tuner_x, model_x_index, model_y_index, points, prm_, T, replace_value)
        err_ = rmse(tuner_y, tuner_p)

        #精度が改善すれば更新
        err_next, prm_next, same_cnt_next, n_next = cond(err_ < err, lambda: [err_, prm_, 0, n], lambda: [err, prm, same_cnt+1, (n+1)%col])

        i_next = i+1
        params_next = [i_next, n_next, err_next, prm_next, same_cnt_next]
        return params_next
    params = [i, n, err, prm, same_cnt]
    params_result = while_loop(lambda params: params[-1] < col, update_loop, params)
    [i_result, n_result, err_result, prm_result, same_cnt_result] = params_result
    return {"prm": prm_result, "err": err_result, "try_cnt": i_result+1}



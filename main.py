import numpy as np
import matplotlib.pyplot as plt
import math, copy


def compute_cost(x, y, w, b):
    cost = 0
    m = x.shape[0]
    for i in range(0, m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

    
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    gradient_w = 0
    gradient_b = 0
    for i in range(0, m):
        f_wb = w * x[i] + b
        gradient_w += x[i] * (f_wb - y[i])
        gradient_b += (f_wb - y[i])
    gradient_w = gradient_w / m
    gradient_b = gradient_b / m
    return gradient_w, gradient_b

def gradientDescent(x, y, w_in, b_in, alpha,num_iters, compute_cost = compute_cost, compute_gradinet = compute_gradient):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        gradient_w, gradient_b  = compute_gradient(x, y, w, b)
        w_temp = w - alpha * gradient_w
        b_temp = b - alpha * gradient_b
        w = w_temp
        b = b_temp
        if i < 10000:
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {gradient_w: 0.3e}, dj_db: {gradient_b: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history
    
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

m = x_train.shape[0]
print(f"Number of training examples is: {m}")

w_init = 100 
b_init = 100
# some gradient descent settings
iterations = 1000
tmp_alpha = 5.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradientDescent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    
    # plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
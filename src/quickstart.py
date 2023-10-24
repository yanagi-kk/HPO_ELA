from pflacco.sampling import create_initial_sample

from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.local_optima_network_features import compute_local_optima_network, calculate_lon_features

# 任意の目的関数
def objective_function(x):
    return x[0]**2 - x[1]**2

dim = 2
# latin hyper cube sampling で初期サンプルを作成する.
X = create_initial_sample(dim, sample_type = 'lhs')
# 任意の目的関数（ここではy = x1^2 - x2^2）を用いて、初期サンプルの目的値を計算する.
y = X.apply(lambda x: objective_function(x), axis = 1)

# flaccoの従来のELA特徴量から、模範的な特徴量を計算する.
ela_distr = calculate_ela_distribution(X, y)
print("ela_distr: ", ela_distr)

# flaccoにまだ含まれていない新しい特徴から、模範的な特徴セットを計算する.
fdc = calculate_fitness_distance_correlation(X, y)
print("fdc: ", fdc)

# ローカルオプティマネットワーク（LON）を計算する. このネットワークから、LONの特徴を算出することができる.
nodes, edges = compute_local_optima_network(f=objective_function, dim=dim, lower_bound=0, upper_bound=1)
lon = calculate_lon_features(nodes, edges)
print("lon: ", lon)

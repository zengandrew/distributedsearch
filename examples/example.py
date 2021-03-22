from distributedsearch import DistributedSearch

def objective_function(rank, scratch_dir, params):
    x = params['x']
    y = (x-3)**2
    return y

if __name__ == '__main__':
    search_space = {
        'x': range(-10, 10)
    }
    
    search = DistributedSearch(search_space, objective_function)
    best_params, min_error = search.get_minimum(trials=50)
    
    print('Best params:', best_params) # Best params: {'x': 3}
    print('Min error:', min_error)     # Min error: 0

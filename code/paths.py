import os

def model_path( algorithm, list_of_parameters, load_saved_models, training ):
    if algorithm == 'fista-ld':
        dataset, dataset_ratio, problem, miniter, maxiter, gamma, alpha, date, ablation = list_of_parameters
        path = os.path.join( os.getcwd(), 'models', 'fista-ld', dataset, str( int( dataset_ratio * 100 ) ) + 'percent', problem )
        if maxiter == miniter:
            path = os.path.join( path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            path = os.path.join( path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        path = os.path.join( path, f'gamma_{gamma:.2e}', f'alpha_{alpha:.2e}', 'date_' + date, 'ablation_' + ablation )
    elif algorithm == 'deepopt':
        dataset, dataset_ratio, problem, miniter, maxiter, alpha, date = list_of_parameters
        path = os.path.join( os.getcwd(), 'models', 'deepopt', dataset, str( int( dataset_ratio * 100 ) ) + 'percent', problem )
        if maxiter == miniter:
            path = os.path.join( path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            path = os.path.join( path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        path = os.path.join( path, 'alpha' + str(alpha)[2:] )
        path = os.path.join( path, 'date_' + date )
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    
    if training:
        if os.path.exists(path) and not load_saved_models:
            for ( dirpath, dirnames, filenames ) in os.walk(path):
                for fi in filenames:
                    os.remove( os.path.join( dirpath, fi ) )
        
        if not os.path.exists(path):
            os.makedirs(path)
    
    return path

def hist_path( algorithm, list_of_parameters ):
    if algorithm == 'ista':
        mode, dataset, dataset_ratio, problem = list_of_parameters
        path = os.path.join( os.getcwd(), 'hist', mode, 'untrained', algorithm, \
                             dataset, str( int( dataset_ratio * 100 ) ) + 'percent', problem )
    elif algorithm == 'fista':
        mode, dataset, dataset_ratio, problem, gamma, alpha, = list_of_parameters
        path = os.path.join( os.getcwd(), 'hist', mode, 'untrained', algorithm, \
                             f'gamma_{gamma:.2e}', f'alpha_{alpha:.2e}', dataset, str( int( dataset_ratio * 100 ) ) + 'percent', problem )
    elif algorithm == 'fista-ld':
        mode, dataset, train_dataset_ratio, test_dataset_ratio, problem, miniter, maxiter, gamma, alpha, date, ablation, epochs = list_of_parameters
        path = os.path.join( os.getcwd(), 'hist', mode, 'trained', 'fista-ld', dataset, \
                             f'{int(100*train_dataset_ratio)}percent_train', \
                             f'{int(100*test_dataset_ratio)}percent_test', \
                             problem )
        if maxiter == miniter:
            path = os.path.join( path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            path = os.path.join( path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        path = os.path.join( path, f'gamma_{gamma:.2e}', f'alpha_{alpha:.2e}', 'date_' + date, 'ablation_' + ablation  )
        path = os.path.join( path, f'epoch_{epochs}' )
    elif algorithm == 'deepopt':
        mode, dataset, train_dataset_ratio, test_dataset_ratio, problem, miniter, maxiter, alpha, date, epochs = list_of_parameters
        path = os.path.join( os.getcwd(), 'hist', mode, 'trained', 'deepopt', dataset,\
                             f'{int(100*train_dataset_ratio)}percent_train', \
                             f'{int(100*test_dataset_ratio)}percent_test', \
                             problem )
        if maxiter == miniter:
            path = os.path.join( path, 'iters_fixed_' + str(maxiter) )
        elif maxiter > miniter:
            path = os.path.join( path, 'iters_random_' + str(miniter) + '_' + str(maxiter) ) 
        else:
            raise ValueError()
        path = os.path.join( path, 'alpha' + str(alpha)[2:] )
        path = os.path.join( path, 'date_' + date )
        path = os.path.join( path, f'epoch_{epochs}' )
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs( os.path.join( path, 'image' ) )
    
    return path

def plot_path( mode, dataset, test_dataset_ratio, problem, date, subfolder = None ):
    if subfolder is None:
        path = os.path.join(os.getcwd(), 'plots', mode, dataset, str( int( 100* test_dataset_ratio ) ) + 'percent', problem, date )
    else:
        path = os.path.join(os.getcwd(), 'plots', mode, dataset, str( int( 100* test_dataset_ratio ) ) + 'percent', problem, date, subfolder )
    if not os.path.exists(path):
        os.makedirs(path)
    return path
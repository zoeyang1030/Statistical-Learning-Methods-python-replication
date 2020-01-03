import numpy as np

def gen_class(n=20, interval=3, var=1, random_seed=None, label_type=0):
    """Generate 2 homoskedastic Gaussian distributions for modeling.

    Parameters
    ----------
    n : int, optional (default=20)
        The number of scatters in each class.

    interval : int or float, optional (default=3)
        The distance between centers of each class.

    var : int or float, optional (default=1)
        The statistical dispersion of each class.

    random_seed : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Returns
    -------
    X : array of shape (n, 2)
        The generated samples.

    y : array of shape (n,)
        The integer labels for scatters in each class.

    Examples
    --------
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, y, ax = generate_random_sample(n=3, interval=2, var=1, random_seed=1)
    >>> X
    array([[-0.37565464, -2.61175641],
            [-2.52817175, -3.07296862],
            [-1.13459237, -4.3015387 ],
            [ 3.74481176,  1.2387931 ],
            [ 2.3190391 ,  1.75062962],
            [ 3.46210794, -0.06014071]])
    >>> y
    array([0., 0., 0., 1., 1., 1.])
    """
    if not random_seed is None: np.random.seed(random_seed)

    x1 = np.random.normal(size=(n, 2), loc=-interval, scale=var)
    x1 += np.random.normal(size=(n, 2), loc=0, scale=var/3)
    x2 = np.random.normal(size=(n, 2), loc=interval, scale=var)
    x2 += np.random.normal(size=(n, 2), loc=0, scale=var/3)
    y1 = -np.ones(n) if label_type else np.zeros(n)
    y2 = np.ones(n)

    X = np.append(x1, x2, axis=0)
    y = np.append(y1, y2)

    return X, y

def split_train_test(X, y, p=0.3, random_seed=None):
    """Split original dataset into training set and testing set.

    Parameters
    ----------
    X : array
        The array X holds the features as columns and samples as rows .

    y : array
        Categories.

    p : float (0<p<1), optional (default=0.3)
        The percentage of testing set in original dataset.

    random_seed : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random.

    Returns
    -------
    X_train : array
        The feature training set.
        
    X_test : array
        The feature testing set.

    y_train : array
        The category testing sets.
        
    y_test : array
        The category testing sets.

    Examples
    --------
    >>> split_train_test(X, y, 0.3, 1)
    (array([[-1.37565464, -3.61175641],
         [-3.52817175, -4.07296862],
         [-2.13459237, -5.3015387 ],
         [-1.25518824, -3.7612069 ],
         [-2.6809609 , -3.24937038],
         [ 2.6775828 ,  2.61594565],
         [ 4.13376944,  1.90010873]]), 
     array([[4.46210794, 0.93985929],
         [2.82757179, 2.12214158],
         [3.04221375, 3.58281521]]), 
     array([-1., -1., -1., -1., -1.,  1.,  1.]), 
     array([1., 1., 1.]))
    """
    if not random_seed is None: np.random.seed(random_seed)
    length = len(y)
    test_index = np.random.choice(range(length), int(length*p), replace=False)
    train_index = np.setdiff1d(np.array(range(length)), test_index)
    return X[train_index], X[test_index], y[train_index], y[test_index]